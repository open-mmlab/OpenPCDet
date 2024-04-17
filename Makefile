.PHONY: build run dev push pull clean

###########################################
# Variables:
###########################################
docker_repo = us-central1-docker.pkg.dev/acquisitions-250323/zendar-docker-repo
docker_image_name = openpcdet
docker_image_tag = zendar
###########################################
# Make commands:
###########################################
build:
	docker build -f docker/cu116.Dockerfile -t $(docker_repo)/$(docker_image_name):$(docker_image_tag) .

run:
	docker run -it --rm --gpus all $(docker_repo)/$(docker_image_name):$(docker_image_tag) bash

dev:  # Run the container with host's `zendar` folder mounted into the container, for development:
	docker run -it --rm --gpus all -v `pwd`/zendar:/OpenPCDet/zendar $(docker_repo)/$(docker_image_name):$(docker_image_tag) bash

push:
	gcloud auth configure-docker us-central1-docker.pkg.dev && \
	docker push $(docker_repo)/$(docker_image_name):$(docker_image_tag)

pull:
	gcloud auth configure-docker us-central1-docker.pkg.dev && \
	docker pull $(docker_repo)/$(docker_image_name):$(docker_image_tag)

clean:
	docker rmi $(docker_repo)/$(docker_image_name):$(docker_image_tag) && \
	docker system prune -af --volumes
