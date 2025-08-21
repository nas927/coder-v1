PYTHON = python3
PIP = pip
VENV_NAME = .venv
REQUIREMENTS = requirements.txt

# Detect OS
IS_WINDOWS := $(findstring Windows_NT,$(OS))

venv:
	@ echo "Creating virtual environment..."
	@ if [ ! -d "$(VENV_NAME)" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
	fi

	@ echo "Installing modules into $(VENV_NAME)..."

	@ if [ "$(IS_WINDOWS)" = "Windows_NT" ]; then \
		$(VENV_NAME)/Scripts/activate.bat && \
		$(PIP) install --upgrade pip && \
		$(PIP) install -r $(REQUIREMENTS); \
	else \
		. $(VENV_NAME)/bin/activate && \
		$(PIP) install --upgrade pip && \
		$(PIP) install -r $(REQUIREMENTS); \
	fi

	@ echo "\033[0;32mTo activate the virtual environment, run:\033[m"
	@ if [ "$(IS_WINDOWS)" = "Windows_NT" ]; then \
		echo "\033[1;34m$(VENV_NAME)\\Scripts\\activate.bat\033[m"; \
	else \
		echo "\033[1;34msource $(VENV_NAME)/bin/activate\033[m"; \
	fi

clean:
	rm -rf $(VENV_NAME)

re: clean venv

.PHONY: venv clean re