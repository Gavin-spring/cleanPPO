from setuptools import setup, find_packages

setup(
    name="ANN_KP",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'setup = setup:main',
            'verify-install = Scripts.verify_installation:verify',
            'check-env = src.env.check_env:main', 
            
            'generate = Scripts.generate_data:main',
            'preprocess = Scripts.preprocess_data:main',
            'train = Scripts.train_model:main',
            'train-sb3 = Scripts.train_sb3:main',
            'evaluate = Scripts.evaluate_solvers:main',
            'evaluate-sb3 = Scripts.evaluate_sb3:main',
        ],
    }
)