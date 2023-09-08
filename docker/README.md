# How-to

## Build

### Lightning-Qubit

```python
TARGET=wheel-lightning-qubit
docker build -f docker/Dockerfile --tag=vincentmichaudrioux/pennylane --target ${TARGET} .
```

## Run

```python
TARGET=wheel-lightning-qubit
docker run -v `pwd`:/io -it ${TARGET} bash
```

## Test

```python
pip install pytest pytest-mock flaky
pl-device-test --device default.qubit
pl-device-test --device default.qubit --shots 10000
pl-device-test --device lightning.qubit
pl-device-test --device lightning.qubit --shots 10000
```