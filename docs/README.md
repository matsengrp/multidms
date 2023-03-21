
# multidms docs

To build the docs make sure you have activated the developer requirements as described in the [main README](../README.md)

```
make clean and make html
xdg-open _build/html/index.html
```

## Deployment to gh-pages

This [gh action](../.github/workflows/docs_pages_workflow.yml) will build the documentation and deploy to gh-pages upon a push to the ``main`` branch of this repository.


