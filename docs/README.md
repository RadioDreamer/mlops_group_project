## Generating the docs

Use [mkdocs](http://www.mkdocs.org/) structure to update the documentation.

Build locally with (recommended):

    uv run invoke build-docs

Serve locally with (recommended):

    uv run invoke serve-docs

If you want to call MkDocs directly, use the config file in this folder:

    uv run mkdocs build --config-file docs/mkdocs.yaml
    uv run mkdocs serve --config-file docs/mkdocs.yaml
