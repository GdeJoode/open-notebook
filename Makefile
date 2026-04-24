.PHONY: run frontend check ruff database lint api start-all stop-all status clean-cache
.PHONY: docker-buildx-prepare docker-buildx-clean docker-buildx-reset
.PHONY: docker-push docker-push-latest docker-release tag export-docs

# Get version from pyproject.toml
VERSION := $(shell grep -m1 version pyproject.toml | cut -d'"' -f2)

# Image names for both registries
DOCKERHUB_IMAGE := lfnovo/open_notebook
GHCR_IMAGE := ghcr.io/lfnovo/open-notebook

# Build platforms
PLATFORMS := linux/amd64,linux/arm64

database:
	docker compose up -d surrealdb

run:
	@echo "⚠️  Warning: Starting frontend only. For full functionality, use 'make start-all'"
	cd frontend && npm run dev

frontend:
	cd frontend && npm run dev

lint:
	uv run python -m mypy .

ruff:
	ruff check . --fix

# === Docker Build Setup ===
docker-buildx-prepare:
	@docker buildx inspect multi-platform-builder >/dev/null 2>&1 || \
		docker buildx create --use --name multi-platform-builder --driver docker-container
	@docker buildx use multi-platform-builder

docker-buildx-clean:
	@echo "🧹 Cleaning up buildx builders..."
	@docker buildx rm multi-platform-builder 2>/dev/null || true
	@docker ps -a | grep buildx_buildkit | awk '{print $$1}' | xargs -r docker rm -f 2>/dev/null || true
	@echo "✅ Buildx cleanup complete!"

docker-buildx-reset: docker-buildx-clean docker-buildx-prepare
	@echo "✅ Buildx reset complete!"

# === Docker Build Targets ===

# Build and push version tags ONLY (no latest)
docker-push: docker-buildx-prepare
	@echo "📤 Building and pushing version $(VERSION) to both registries..."
	docker buildx build --pull \
		--platform $(PLATFORMS) \
		--progress=plain \
		-t $(DOCKERHUB_IMAGE):$(VERSION) \
		-t $(GHCR_IMAGE):$(VERSION) \
		--push \
		.
	@echo "✅ Pushed version $(VERSION) to both registries (latest NOT updated)"

# Update v1-latest tags to current version
docker-push-latest: docker-buildx-prepare
	@echo "📤 Updating v1-latest tags to version $(VERSION)..."
	docker buildx build --pull \
		--platform $(PLATFORMS) \
		--progress=plain \
		-t $(DOCKERHUB_IMAGE):$(VERSION) \
		-t $(DOCKERHUB_IMAGE):v1-latest \
		-t $(GHCR_IMAGE):$(VERSION) \
		-t $(GHCR_IMAGE):v1-latest \
		--push \
		.
	@echo "✅ Updated v1-latest to version $(VERSION)"

# Full release: push version AND update latest tags
docker-release: docker-push-latest
	@echo "✅ Full release complete for version $(VERSION)"

tag:
	@version=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "Creating tag v$$version"; \
	git tag "v$$version"; \
	git push origin "v$$version"


api:
	uv run app-main

# === Service Management ===
start-all:
	@echo "Starting Open Notebook (Database + API + Frontend)..."
	@echo "Starting SurrealDB..."
	@docker compose up -d surrealdb
	@echo "Waiting for SurrealDB..."
	@sleep 3
	@echo "Starting API backend (includes background worker)..."
	@uv run app-main &
	@sleep 2
	@echo "Starting Next.js frontend..."
	@echo ""
	@echo "All services started!"
	@echo "  Frontend: http://localhost:3000"
	@echo "  API:      http://localhost:5055"
	@echo "  API Docs: http://localhost:5055/docs"
	@echo ""
	cd frontend && npm run dev

stop-all:
	@echo "Stopping all Open Notebook services..."
	@pkill -INT -f "next dev" || true
	@pkill -INT -f "app_main.api.app" || true
	@sleep 2
	@pkill -f "app_main.api.app" || true
	@docker compose down
	@echo "All services stopped."

status:
	@echo "Open Notebook Service Status:"
	@echo "  Database (SurrealDB):"
	@docker compose ps surrealdb 2>/dev/null || echo "    Not running"
	@echo "  API Backend + Worker:"
	@pgrep -f "app_main.api.app" >/dev/null && echo "    Running" || echo "    Not running"
	@echo "  Next.js Frontend:"
	@pgrep -f "next dev" >/dev/null && echo "    Running" || echo "    Not running"

# === Documentation Export ===
export-docs:
	@echo "📚 Exporting documentation..."
	@uv run python scripts/export_docs.py
	@echo "✅ Documentation export complete!"

# === Cleanup ===
clean-cache:
	@echo "🧹 Cleaning cache directories..."
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -type f -delete 2>/dev/null || true
	@find . -name "*.pyo" -type f -delete 2>/dev/null || true
	@find . -name "*.pyd" -type f -delete 2>/dev/null || true
	@echo "✅ Cache directories cleaned!"