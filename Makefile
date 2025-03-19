setup_env:
	pyenv virtualenv 3.10.6 stockify_env
	pyenv local stockify_env
	touch .env .envrc
	echo "dotenv" >> .envrc
	direnv allow

# Set GCP environment variables
set_gcp:
	@echo "Setting GCP Environment Variables..."
	@echo 'export GCP_PROJECT_ID="lewagon-statistical-arbitrage"' > gcp_env.sh
	@echo 'export DOCKER_IMAGE_NAME="stat-api-interface"' >> gcp_env.sh
	@echo 'export GCP_REGION="europe-west1"' >> gcp_env.sh
	@echo 'export DOCKER_REPO_NAME="stat-replica-lewagon"' >> gcp_env.sh
	@echo "Run 'source gcp_env.sh' to apply environment variables"

make_repo:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev
	gcloud artifacts repositories create $(DOCKER_REPO_NAME) --repository-format=docker --location=$(GCP_REGION) --description=""


# Docker Build
build:
	docker build -t $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME):0.1 .

# Docker Push
push:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME):0.1

# Deploy to Cloud Run
deploy:
	gcloud run deploy --image $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME):0.1 --region $(GCP_REGION)

# Full Pipeline: Set env, build, push, deploy
full_deploy: set_gcp build push deploy
