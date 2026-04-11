# 车辆故障预测系统安装配置
from setuptools import setup, find_packages

setup(
    name="vehicle_fault_prediction",
    version="1.0.0",
    description="工业级智能车辆故障预测系统",
    author="ML DeepWay Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "pydantic",
        "optuna",
        "scipy",
    ],
    extras_require={
        "full": [
            "python-can",
            "paho-mqtt",
            "obd",
            "influxdb-client",
            "onnx",
            "onnxruntime",
            "tensorboard",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
