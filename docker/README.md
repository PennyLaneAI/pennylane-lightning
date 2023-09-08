# How-to

## Download

```shell
docker run -v pwd:/io -it vincentmichaudrioux/pennylane bash
```

## Build

### Lightning-Qubit

```shell
TARGET=wheel-lightning-qubit
docker build -f docker/Dockerfile --tag=${TARGET} --target ${TARGET} .
```

## Run

```shell
TARGET=wheel-lightning-qubit
docker run -v `pwd`:/io -it ${TARGET} bash
```

## Test

```shell
pip install pytest pytest-mock flaky
pl-device-test --device default.qubit
pl-device-test --device default.qubit --shots 10000
pl-device-test --device lightning.qubit
pl-device-test --device lightning.qubit --shots 10000
```