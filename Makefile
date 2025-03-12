setup_env:
	pyenv virtualenv 3.10.6 stockify_env
	pyenv local stockify_env
	touch .env .envrc
	echo "dotenv" >> .envrc
	direnv allow
