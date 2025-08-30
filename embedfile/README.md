## embedfile - CLI tool for embedding and searching text data using SQLite and LLMs

### SYNOPSIS
    embedfile [OPTIONS] COMMAND [ARGS...]

### DESCRIPTION
    embedfile is a high-performance command-line tool for generating,
    storing, and querying vector embeddings from structured or unstructured
    text data. It leverages llamafile, sqlite-lembed, sqlite-vec, and SQLite
    to create an embedded semantic search engine with zero external dependencies.

### OPTIONS
    -m, --model FILE
        Specify the path to the embedding model (.gguf format).

    -v, --version
        Show version information for embedfile and dependencies.

    -h, --help
        Display help message and usage information.

### COMMANDS
    embedfile embed [TEXT]
        Generate a JSON vector embedding for the input string.
        If TEXT is omitted, reads from standard input line-by-line and emits embeddings.

    embedfile import [--embed COLUMN] [--table NAME] SOURCE_FILE INDEX_DB
        Import a structured file (CSV, JSON, NDJSON, or TXT) or SQLite .db file into a SQLite
        database and embed the specified column. If the source is a TXT file,
        embedding is done on each line.

        Options:
            --embed COLUMN
                The name of the column to embed (required for all formats except TXT).

            --table NAME or -t NAME
                Specify table name when importing from a SQLite .db file.

    embedfile search [--k NUM] INDEX_DB QUERY
        Search the embedded SQLite database using the specified query string
        and return top NUM (default: 10) semantically similar results.

    embedfile sh [INDEX_DB]
        Launch an interactive SQLite shell with all relevant extensions preloaded.

### EXAMPLES
    Embed a string:
        embedfile embed "hello world"

    Embed from standard input:
        echo "Paris is the capital of France." | embedfile embed

    Import and embed data from a CSV file:
        embedfile --model ./model.gguf import --embed description products.csv products.db

    Search for similar entries:
        embedfile search products.db "wireless headphones"

    Launch interactive shell:
        embedfile sh
        embedfile sh < commands.sql

### AUTHOR
    Created by Alex Garcia.
