LINE_WIDTH=99
ISORT_FLAGS=--line-width=${LINE_WIDTH} --profile black
BLACK_FLAGS=--skip-string-normalization --line-length=${LINE_WIDTH}
PYTEST_FLAGS=-p no:warnings

install:
	pip install -r requirements.txt

format:
	isort ${ISORT_FLAGS} --check-only --diff .
	black ${BLACK_FLAGS} --check --diff .

format-fix:
	isort ${ISORT_FLAGS} .
	black ${BLACK_FLAGS} .
