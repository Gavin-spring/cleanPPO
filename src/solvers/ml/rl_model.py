# ref: https://github.com/pemami4911/neural-combinatorial-rl-pytorch
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

# from beam_search import Beam


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""
    def __init__(self, embedding_dim, hidden_dim, use_cuda):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.use_cuda = use_cuda
        self.enc_init_state = self.init_hidden(hidden_dim) # this is not used in the forward pass, but can be

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden
    
    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if self.use_cuda:
            enc_init_hx = enc_init_hx.cuda()

        enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if self.use_cuda:
            enc_init_cx = enc_init_cx.cuda()

        return (enc_init_hx, enc_init_cx)

class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=False, C=10, use_cuda=True):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()
        
        v = torch.FloatTensor(dim)
        if use_cuda:
            v = v.cuda()  
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(dim)) , 1. / math.sqrt(dim))
        
    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits

class Decoder(nn.Module):
    """
    The Decoder module for the Pointer Network, correctly adapted for the
    Knapsack problem. It uses an augmented query (decoder state + capacity
    embedding) to drive the attention mechanism at each step.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                #  max_length: int,
                 tanh_exploration: float,
                 use_tanh: bool,
                 n_glimpses: int = 1,
                 use_cuda: bool = True):
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.max_length = max_length
        self.use_cuda = use_cuda
        self.n_glimpses = n_glimpses
        
        # This will be set externally by RLSolver before training or evaluation
        self.decode_type = "stochastic"

        # Attention modules for pointing and Glimpsing
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, use_cuda=self.use_cuda)
        self.glimpse = Attention(hidden_dim, use_tanh=False, use_cuda=self.use_cuda)
        
        self.sm = nn.Softmax(dim=1)
        
        # Use an LSTMCell for clean, single-step recurrent logic
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)

        # --- NEW: Layers for capacity-aware query ---
        # 1. A small embedding layer for the scalar capacity value.
        # It projects the capacity from a single scalar to a small vector.
        self.capacity_embedding_dim = 16  # This can be a tunable hyperparameter
        self.capacity_embedder = nn.Linear(1, self.capacity_embedding_dim)

        # 2. A projection layer for the augmented query.
        # It takes the concatenated vector [decoder_hidden_state, capacity_embedding]
        # and projects it back to the model's hidden_dim.
        self.query_projection = nn.Linear(hidden_dim + self.capacity_embedding_dim, hidden_dim)

    def apply_mask_to_logits(self, logits, mask):
        """Applies a mask to the logits, preventing re-selection of items."""
        if mask is None:
            return logits # No mask on the first step
        
        # Set logits of masked (already selected) items to a large negative number
        # out-of-place, so we don't modify the original logits tensor.
        # torch.where(condition, value_if_true, value_if_false)
        # where the mask equals true is where we want to apply the large negative value.
        masked_logits = torch.where(mask, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device), logits)
        return masked_logits

    def forward(self, decoder_input, embedded_inputs, hidden, context, capacity, attention_mask, weights):
        """
        The main forward pass for the decoder.
        Args:
            decoder_input: The initial input ('start symbol'). Shape: [batch_size, embedding_dim]
            embedded_inputs: Embedded item features. Shape: [seq_len, batch_size, embedding_dim]
            hidden: Initial hidden state from encoder. Tuple (h_0, c_0)
            context: Encoder outputs for attention. Shape: [seq_len, batch_size, hidden_dim]
            capacity: The knapsack capacity. Shape: [batch_size]
            attention_mask: The padding mask. Shape: [batch_size, seq_len]
        """
        # --- Prepare capacity embedding once before the loop ---
        # Reshape capacity from (batch_size) to (batch_size, 1) for the linear layer
        capacity_unsqueezed = capacity.unsqueeze(1)
        capacity_embedded = self.capacity_embedder(capacity_unsqueezed)

        # Ensure capacity_embedded has the shape [batch_size, capacity_embedding_dim]
        remaining_capacity = capacity.clone()

        # Setup for the decoding loop dynamically
        batch_size = context.size(1)
        seq_len = context.size(0)
        
        outputs = []
        selections = []
        # Initialize a mask to track selected items
        selection_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=context.device)

        # --- Iterate for the maximum decoding length ---
        for _ in range(seq_len):
            hx, cx = self.lstm_cell(decoder_input, hidden)
            hidden = (hx, cx)
            # Concatenate the current decoder state with the capacity embedding
            augmented_query = torch.cat([hx, capacity_embedded], dim=1)
            # Project the augmented query to the correct dimension for attention
            projected_query = self.query_projection(augmented_query)

            # Glimpse Mechanism (uses the capacity-aware query)
            g_l = projected_query 
            for _ in range(self.n_glimpses):
                _, glimpse_logits = self.glimpse(g_l, context)
                # Glimpse logits are masked to prevent attention on padding
                # out-of-place operation, compiler friendly
                glimpse_logits = torch.where(~attention_mask, torch.tensor(-1e9, dtype=glimpse_logits.dtype, device=glimpse_logits.device), glimpse_logits)

                glimpse_weights = self.sm(glimpse_logits)
                g_l = torch.bmm(context.permute(1, 2, 0), glimpse_weights.unsqueeze(2)).squeeze(2)

            # Final Pointer (also uses the capacity-aware query)
            final_augmented_query = torch.cat([g_l, capacity_embedded], dim=1)
            final_projected_query = self.query_projection(final_augmented_query)

            _, pointer_logits = self.pointer(final_projected_query, context)
            
            # Apply masks to pointer logits:
            repetition_mask = selection_mask
            capacity_mask = weights <= remaining_capacity.unsqueeze(1)
            combined_mask = attention_mask & ~repetition_mask & capacity_mask

            masked_logits = torch.where(combined_mask, pointer_logits, torch.tensor(-1e9, dtype=pointer_logits.dtype, device=pointer_logits.device))

            # Convert masked logits to float for stability
            stable_logits = masked_logits.float()
            probs = self.sm(stable_logits)

            # find rows with all NaN probabilities in the batch
            nan_rows = torch.isnan(probs).all(dim=1)
            if torch.any(nan_rows):
                # replace NaN rows with uniform probabilities
                # avoid breakdown in multinomial, and have reasonable small influence on final results
                num_valid_items = attention_mask.sum(dim=1).float()
                uniform_probs = attention_mask.float() / num_valid_items.unsqueeze(1).clamp(min=1)
                # out-of-place operation, do not modify the original probs tensor when gradient computation
                probs = torch.where(nan_rows.unsqueeze(1), uniform_probs, probs)

            # Select the next item based on the decoding strategy
            if self.decode_type == "stochastic":
                idxs = self.decode_stochastic(probs)
            elif self.decode_type == "greedy":
                idxs = self.decode_greedy(probs)
            else:
                raise ValueError(f"Unknown decode_type: {self.decode_type}")

            # Prepare for the next decoding step
            decoder_input = embedded_inputs.gather(0, idxs.view(1, -1, 1).expand(1, -1, self.embedding_dim)).squeeze(0)

            outputs.append(probs)
            selections.append(idxs)

            # Update the mask for the next iteration
            selection_mask = selection_mask.scatter(1, idxs.unsqueeze(1), True)
            
            chosen_weights = weights.gather(1, idxs.unsqueeze(1)).squeeze(1)
            is_valid_choice = combined_mask.gather(1, idxs.unsqueeze(1)).squeeze(1)
            remaining_capacity -= chosen_weights * is_valid_choice.float()

        return outputs, selections

    def decode_stochastic(self, probs):
        """Sample from the probability distribution."""
        # To keep data stablility when AMP and torch compile
        # before torch.multinomial, forecefully convert probabilities tensor to float
        stable_probs = probs.float()
        return torch.multinomial(stable_probs, 1).squeeze(1)

    def decode_greedy(self, probs):
        """Select the item with the highest probability."""
        return torch.argmax(probs, dim=1)

class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq model"""
    def __init__(self, config, use_cuda, input_dim=2):
        super(PointerNetwork, self).__init__()

        embedding_dim = config.hyperparams.embedding_dim
        hidden_dim = config.hyperparams.hidden_dim
        n_glimpses = config.hyperparams.n_glimpses
        tanh_exploration = config.hyperparams.tanh_exploration
        use_tanh = config.hyperparams.use_tanh

        # This layer will project the raw 2D input (weight, value) to the embedding_dim (128)
        self.embedding = nn.Linear(input_dim, embedding_dim)

        self.encoder = Encoder(
                embedding_dim, # Note: Encoder's input_dim is the embedding_dim
                hidden_dim,
                use_cuda)

        self.decoder = Decoder(
                embedding_dim,
                hidden_dim,
                # max_length=max_decoding_len,
                tanh_exploration=tanh_exploration,
                use_tanh=use_tanh,
                n_glimpses=n_glimpses,
                use_cuda=use_cuda)

        # Trainable initial hidden states
        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.decoder_in_0.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                                       1. / math.sqrt(embedding_dim))

        # Value head for the value prediction
        # This is a temporary solution and would be replaced with a real Critic later
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
            
    def forward(self, inputs, capacity, attention_mask):
        """
        Propagate inputs through the network.
        Args:
            inputs: The input tensor of items. Shape: [batch_size, seq_len, input_dim]
            capacity: The knapsack capacity. Shape: [batch_size]
        """
        batch_size = inputs.size(0)

        # 1. Embed inputs
        # Shape becomes: [batch_size, seq_len, embedding_dim]
        embedded_inputs = self.embedding(inputs)

        # 2. Permute for LSTM
        # For LSTM, inputs must be [seq_len, batch_size, embedding_dim].
        embedded_inputs_for_encoder = embedded_inputs.permute(1, 0, 2)
        
        # --- FIX IS HERE ---
        # Your Encoder's forward method requires an explicit hidden state.
        # We create a zero-initialized hidden state to pass to it.
        # The shape must be (num_layers, batch_size, hidden_dim). We assume num_layers=1.
        initial_hidden = (torch.zeros(1, batch_size, self.encoder.hidden_dim, device=inputs.device),
                        torch.zeros(1, batch_size, self.encoder.hidden_dim, device=inputs.device))
        
        # 3. Encode inputs, now passing the initial_hidden state
        # This call now matches your Encoder's forward(self, inputs, hidden) signature.
        context, encoder_hidden = self.encoder(embedded_inputs_for_encoder, initial_hidden)
        
        # 4. Squeeze the hidden state for the LSTMCell in the Decoder
        # This is still necessary to fix the 3D vs 2D shape mismatch for the Decoder.
        decoder_init_state = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))
        
        # 5. Prepare initial input for the decoder
        # Note: I'm using self.decoder_in_0 from your original code.
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(batch_size, 1)
        
        # 6. Decode (point to items)
        # The permuted 'context' is correct for attention.
        # embedded_inputs needs to be passed for the gather operation inside the decoder.
        weights = inputs.select(2, 0) # weights are the first feature
        (probs_list, actions_list) = self.decoder(decoder_input,
                                                embedded_inputs_for_encoder, # Pass permuted version
                                                decoder_init_state,
                                                context,
                                                capacity,
                                                attention_mask,
                                                weights)

        # # 7. Stack the lists of outputs into tensors
        # # probs = torch.stack(probs_list, dim=1)
        # action_idxs = torch.stack(actions_list, dim=0)

        # return probs_list, action_idxs
        
        # --- 使用Critic头计算状态价值 ---
        # 我们用encoder最终的隐状态作为整个状态的表征
        # encoder_hidden[0] 的 shape 是 [num_layers, batch_size, hidden_dim]
        # 我们取最后一层的隐状态
        last_hidden_state = encoder_hidden[0][-1] # Shape: [batch_size, hidden_dim]
        state_value = self.value_head(last_hidden_state).squeeze(-1) # Shape: [batch_size]

        # --- 修改返回值 ---
        # 返回PPO需要的所有信息
        return probs_list, actions_list, state_value
