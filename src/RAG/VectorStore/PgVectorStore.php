<?php

namespace NeuronAI\RAG\VectorStore;

use NeuronAI\RAG\Document;
use PgSql\Connection;
use Pgvector\Vector;

class PgVectorStore implements VectorStoreInterface
{

    public function __construct(
        protected Connection $connection,
        protected string $table,
        protected string $indexName,
        protected int $vectorDimension,
        protected bool $createTable = true,
        protected bool $dropTable = false
    ) {
        // Ensure the pgvector extension is installed
        pg_query($this->connection, "CREATE EXTENSION IF NOT EXISTS vector");

        // Should we drop the table?
        if ($this->dropTable) {
            pg_query($this->connection, "DROP TABLE IF EXISTS {$this->table}");
        }

        // Should we create the table?
        if ($this->createTable) {
            $this->createTable();
        }

    }

    protected function createTable(): void
    {
        try {

            pg_query($this->connection, 'BEGIN');

            // Create the table if it doesn't exist
            pg_query($this->connection, "CREATE TABLE IF NOT EXISTS {$this->table} (
                id SERIAL PRIMARY KEY,
                index_name TEXT NOT NULL,
                content TEXT,
                embedding VECTOR({$this->vectorDimension}),
                metadata JSONB DEFAULT '{}'::jsonb
            )");

            // Create the index on index_name if it doesn't exist
            pg_query($this->connection, "CREATE INDEX IF NOT EXISTS {$this->table}_index_name_idx ON {$this->table} (index_name)");

            pg_query($this->connection, "COMMIT");

        } catch (\Throwable $e) {
            pg_query($this->connection, 'ROLLBACK');
            throw $e;
        }
    }

    public function addDocument(Document $document): void
    {
        if ($document->embedding === null) {
            throw new \Exception('document embedding must be set before adding a document');
        }

        $this->checkVectorDimension(count($document->embedding));

        try{

            pg_query($this->connection, 'BEGIN');

            $embedding = new Vector($document->embedding);

            pg_query_params(
                $this->connection,
                "INSERT INTO {$this->table} (index_name, content, embedding, metadata) VALUES ($1, $2, $3, $4)",
                [
                    $this->indexName,
                    $document->content,
                    $embedding,
                    json_encode($document->metadata) ?? '{}'
                ]
            );

            pg_query($this->connection, 'COMMIT');

        } catch (\Throwable $e) {
            pg_query($this->connection, 'ROLLBACK');
            throw $e;
        }
    }

    /**
     * @param  array<Document>  $documents
     */
    public function addDocuments(array $documents): void
    {
        if (empty($documents)) {
            return;
        }

        if ($documents[0]->embedding === null) {
            throw new \Exception('document embedding must be set before adding a document');
        }

        $this->checkVectorDimension(count($documents[0]->embedding));

        $values = [];
        $params = [];
        $paramIndex = 1;

        foreach ($documents as $document) {
            if ($document->embedding === null) {
                throw new \Exception('Each document must have an embedding');
            }

            $values[] = '('
                . '$' . $paramIndex . ', '
                . '$' . ($paramIndex + 1) . ', '
                . '$' . ($paramIndex + 2) . ', '
                . '$' . ($paramIndex + 3) . ')';

            $params[] = $this->indexName;
            $params[] = $document->content;
            $params[] = new \Pgvector\Vector($document->embedding);
            $params[] = json_encode($document->metadata ?? []);

            $paramIndex += 4;
        }

        $sql = "INSERT INTO {$this->table} (index_name, content, embedding, metadata) VALUES " . implode(', ', $values);

        try {
            pg_query($this->connection, 'BEGIN');

            $result = pg_query_params($this->connection, $sql, $params);

            pg_query($this->connection, 'COMMIT');

            if ($result === false) {
                throw new \Exception("Batch insert failed: " . pg_last_error($this->connection));
            }

        } catch (\Throwable $e) {
            pg_query($this->connection, 'ROLLBACK');
            throw $e;
        }
    }

    /**
     * Return docs most similar to the embedding.
     *
     * @param  float[]  $embedding
     * @return array<Document>
     */
    public function similaritySearch(array $embedding, int $k = 4): iterable
    {

        $this->checkVectorDimension(count($embedding));

        $result = pg_query_params(
            $this->connection,
            "   SELECT content, metadata
                FROM {$this->table}
                WHERE index_name = $1
                ORDER BY embedding <-> $2
                LIMIT $3",
            [
                $this->indexName,
                new Vector($embedding),
                $k
            ]
        );

        if ($result === false) {
            throw new \Exception("Failed to perform similarity search");
        }

        $documents = [];
        while ($row = pg_fetch_assoc($result)) {
            $document = new Document();
            $document->content = $row['content'];
            $document->metadata = json_decode($row['metadata'], true);
            $documents[] = $document;
        }

        return $documents;
    }

    private function checkVectorDimension(int $dimension): void
    {
        $result = pg_query(
            $this->connection,
            "SELECT COUNT(*) FROM {$this->table} WHERE vector_dims(embedding) != {$dimension}"
        );

        if ($result === false) {
            throw new \Exception("Failed to check vector dimension");
        }

        $row = pg_fetch_row($result);
        if ((int)$row[0] > 0) {
            throw new \Exception("Vector dimension mismatch: some rows do not match expected dimension {$dimension}");
        }
    }
}
