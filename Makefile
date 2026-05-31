# -----------------------------
# Project config (override as needed)
# -----------------------------
MILL             ?= mill
SCALA_VERSION    ?= 3.8.2
WATCH            ?= -w          # set to empty to disable mill watch mode

# Module coordinates
CORE_JVM         := palladium.jvm["$(SCALA_VERSION)"]
CORE_JS          := palladium.js["$(SCALA_VERSION)"]
CORE_NATIVE      := palladium.native["$(SCALA_VERSION)"]
WEB_FRONTEND     := web-frontend
WEB_SERVER       := web-server

# JS output folders produced by mill
JS_OUT_FAST      := out/web-frontend/fastLinkJS.dest
JS_OUT_FULL      := out/web-frontend/fullLinkJS.dest

# Where we symlink the generated JS sources for Vite to pick up
WEB_LINK_TARGET  := web-frontend/web/lib/scala

# -----------------------------
# Phony
# -----------------------------
.PHONY: help bsp ide-setup show-config \
        compile compile-js compile-native compile-server test test-only test-only-w test-js test-native test-native-w \
        js-fast js-full js-watch-fast \
        link-fast link-full \
        web-install web-dev web-build \
        server-run \
        benchmark benchmark-native \
        dev-all clean web-clean

# -----------------------------
# Help
# -----------------------------
help:
	@echo ""
	@echo "Targets:"
	@echo "  show-config         - Print resolved settings."
	@echo "  bsp                 - Install Mill BSP (IDE integration)."
	@echo "  ide-setup           - Install BSP."
	@echo ""
	@echo "Build:"
	@echo "  compile             - Compile core (JVM + JS) + web-frontend + web-server."
	@echo "  compile-js          - Compile web-frontend ScalaJS module only."
	@echo "  compile-server      - Compile web-server JVM module only."
	@echo "  test                - Run all tests (JVM + JS)."
	@echo "  test-only T=X       - Run JVM tests matching X (e.g. T=EinDsl)."
	@echo "  test-only-w T=X     - Same as test-only but with watch mode."
	@echo "  test-native-w       - Run Native tests with watch mode."
	@echo "  test-js             - Run JS platform tests only."
	@echo "  benchmark           - Run JVM benchmarks (compiled vs interpreted)."
	@echo "  benchmark-native    - Run Native benchmarks (compiled vs BLAS vs interpreted)."
	@echo ""
	@echo "Scala.js:"
	@echo "  js-fast             - One-off fastLinkJS (dev JS bundle)."
	@echo "  js-full             - One-off fullLinkJS (optimized prod JS bundle)."
	@echo "  js-watch-fast       - Watch fastLinkJS (rebuild on changes)."
	@echo ""
	@echo "Linking:"
	@echo "  link-fast           - Build fastLinkJS and symlink into Vite project."
	@echo "  link-full           - Build fullLinkJS and symlink into Vite project."
	@echo ""
	@echo "Web (Vite):"
	@echo "  web-install         - npm install (in web-frontend/web)."
	@echo "  web-dev             - Run Vite dev server."
	@echo "  web-build           - Production build (vite build)."
	@echo ""
	@echo "Server:"
	@echo "  server-run          - Start the Tapir/Netty web server."
	@echo ""
	@echo "Workflow:"
	@echo "  dev-all             - Full dev setup: compile, link, install, dev server."
	@echo ""
	@echo "Housekeeping:"
	@echo "  web-clean           - Remove the symlinked $(WEB_LINK_TARGET)."
	@echo "  clean               - Remove mill out/ and link target."
	@echo ""
	@echo "Usage examples:"
	@echo "  make link-fast web-install web-dev"
	@echo "  make dev-all"
	@echo "  make server-run          # in a separate terminal"
	@echo "  make js-watch-fast       # in a separate terminal for live rebuild"
	@echo ""

show-config:
	@echo "MILL             = $(MILL)"
	@echo "SCALA_VERSION    = $(SCALA_VERSION)"
	@echo "WATCH            = $(WATCH)"
	@echo "CORE_JVM         = $(CORE_JVM)"
	@echo "CORE_JS          = $(CORE_JS)"
	@echo "WEB_FRONTEND     = $(WEB_FRONTEND)"
	@echo "WEB_SERVER       = $(WEB_SERVER)"
	@echo "JS_OUT_FAST      = $(JS_OUT_FAST)"
	@echo "JS_OUT_FULL      = $(JS_OUT_FULL)"
	@echo "WEB_LINK_TARGET  = $(WEB_LINK_TARGET)"

# -----------------------------
# IDE
# -----------------------------
bsp:
	$(MILL) --bsp-install

ide-setup: bsp

# -----------------------------
# Compilation
# -----------------------------
compile:
	$(MILL) $(CORE_JVM).compile
	$(MILL) $(CORE_JS).compile
	$(MILL) $(WEB_FRONTEND).compile
	$(MILL) $(WEB_SERVER).compile

compile-js:
	$(MILL) $(WEB_FRONTEND).compile

compile-server:
	$(MILL) $(WEB_SERVER).compile

# -----------------------------
# Tests
# -----------------------------
test:
	$(MILL) $(CORE_JVM).test
	$(MILL) $(CORE_JS).test

test-only:
	@test -n "$(T)" || { echo "Usage: make test-only T=EinDsl"; exit 1; }
	$(MILL) $(CORE_JVM).test -- "*$(T)*"

test-only-w:
	@test -n "$(T)" || { echo "Usage: make test-only-w T=EinDsl"; exit 1; }
	$(MILL) -w $(CORE_JVM).test -- "*$(T)*"

test-js:
	$(MILL) $(CORE_JS).test

compile-native:
	$(MILL) $(CORE_NATIVE).compile

test-native:
	$(MILL) $(CORE_NATIVE).test

test-native-w:
	$(MILL) -w $(CORE_NATIVE).test

benchmark:
	$(MILL) benchmark.run

benchmark-native:
	$(MILL) benchmark-native.run

# -----------------------------
# Scala.js linking
# -----------------------------
js-fast:
	$(MILL) $(WEB_FRONTEND).fastLinkJS

js-full:
	$(MILL) $(WEB_FRONTEND).fullLinkJS

js-watch-fast:
	$(MILL) $(WATCH) $(WEB_FRONTEND).fastLinkJS

# -----------------------------
# Symlinks into web app
# -----------------------------
link-fast: js-fast
	@mkdir -p web-frontend/web/lib
	@test ! -e "$(WEB_LINK_TARGET)" || rm -rf "$(WEB_LINK_TARGET)"
	ln -sfn "$$(realpath "$(JS_OUT_FAST)")" "$(WEB_LINK_TARGET)"
	@echo "Linked: $(WEB_LINK_TARGET) -> $$(realpath "$(JS_OUT_FAST)")"

link-full: js-full
	@mkdir -p web-frontend/web/lib
	@test ! -e "$(WEB_LINK_TARGET)" || rm -rf "$(WEB_LINK_TARGET)"
	ln -sfn "$$(realpath "$(JS_OUT_FULL)")" "$(WEB_LINK_TARGET)"
	@echo "Linked: $(WEB_LINK_TARGET) -> $$(realpath "$(JS_OUT_FULL)")"

# -----------------------------
# Web (Vite)
# -----------------------------
web-install:
	cd web-frontend/web && npm install

web-dev:
	cd web-frontend/web && npm run dev

web-build:
	cd web-frontend/web && npm run build

# -----------------------------
# Server
# -----------------------------
server-run:
	$(MILL) $(WEB_SERVER).run

# -----------------------------
# Full dev workflow
# -----------------------------
dev-all: compile link-fast web-install
	@echo ""
	@echo "Ready! Run in separate terminals:"
	@echo "  make web-dev             # Vite dev server"
	@echo "  make server-run          # API server on :8080"
	@echo "  make js-watch-fast       # Auto-rebuild ScalaJS on changes"
	@echo ""

# -----------------------------
# Cleaning
# -----------------------------
web-clean:
	@test ! -e "$(WEB_LINK_TARGET)" || rm -rf "$(WEB_LINK_TARGET)"
	@echo "Removed link/dir: $(WEB_LINK_TARGET)"

clean: web-clean
	@test ! -e out || rm -rf out
	@echo "Removed: out/"
