from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="gridworld-rl",
    version="1.0.0",
    author="Your Name",
    author_email="you@example.com",
    description="A Grid World reinforcement learning environment with Q-Learning, Value Iteration, and Policy Iteration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gridworld-rl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gridworld-train=examples.train_qlearning:main",
        ],
    },
)
