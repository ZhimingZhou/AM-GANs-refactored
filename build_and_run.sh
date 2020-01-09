DOCKER_FILE=${1:-"Dockerfile"}
IMAGE_NAME=${2:-"am-gan"}

docker build --tag ${IMAGE_NAME} --file ${DOCKER_FILE} .

docker run --gpus 0 --rm -it --user $(id -u):$(id -g) --mount type=bind,source=$(pwd),target=/workspace ${IMAGE_NAME} bash
