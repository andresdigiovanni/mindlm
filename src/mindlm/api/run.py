import uvicorn


def main() -> None:  # pragma: no cover
    uvicorn.run(
        "mindlm.api.main:app",
        host="0.0.0.0",  # noqa: S104
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
