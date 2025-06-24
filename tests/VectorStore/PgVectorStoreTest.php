<?php

declare(strict_types=1);

namespace NeuronAI\Tests\VectorStore;

use NeuronAI\RAG\Document;
use NeuronAI\RAG\VectorStore\PgVectorStore;
use NeuronAI\RAG\VectorStore\VectorStoreInterface;
use PHPUnit\Framework\TestCase;

class PgVectorStoreTest extends TestCase
{
    protected \PgSql\Connection $connection;

    protected array $embedding;

    protected string $table = 'test_pgvector';

    protected function setUp(): void
    {
        if (!$this->isPortOpen('127.0.0.1', 5432)) {
            $this->markTestSkipped('PostgreSQL is not running on port 5432.');
        }

        $this->connection = \pg_connect("host=127.0.0.1 port=5432 dbname=mydatabase user=myuser password=mypassword");

        if (!$this->connection) {
            $this->markTestSkipped('Could not connect to PostgreSQL database.');
        }

        $this->embedding = json_decode(file_get_contents(__DIR__ . '/../stubs/hello-world.embeddings'), true);

        // Clean test table
        @pg_query($this->connection, "DROP TABLE IF EXISTS {$this->table}");
    }

    private function isPortOpen(string $host, int $port, int $timeout = 1): bool
    {
        $connection = @fsockopen($host, $port, $errno, $errstr, $timeout);
        if (is_resource($connection)) {
            fclose($connection);
            return true;
        }
        return false;
    }

    public function test_pgvector_instance(): void
    {
        $store = new PgVectorStore($this->connection, $this->table, 'embedding_index', count($this->embedding));
        $this->assertInstanceOf(VectorStoreInterface::class, $store);
    }

    public function test_add_document_and_search(): void
    {
        $store = new PgVectorStore($this->connection, $this->table, 'embedding_index', count($this->embedding));

        $document = new Document('Hello World!');
        $document->embedding = $this->embedding;

        $store->addDocument($document);

        $results = $store->similaritySearch($this->embedding);
        $this->assertIsIterable($results);
        $this->assertNotEmpty($results);
        $this->assertInstanceOf(Document::class, $results[0]);
    }

    public function test_add_document_with_metadata(): void
    {
        $store = new PgVectorStore($this->connection, $this->table, 'embedding_index', count($this->embedding));

        $document = new Document('Hello World!');
        $document->embedding = $this->embedding;
        $document->metadata = ['key' => 'value'];

        $store->addDocument($document);

        $results = $store->similaritySearch($this->embedding);
        $this->assertIsIterable($results);
        $this->assertNotEmpty($results);
        $this->assertInstanceOf(Document::class, $results[0]);
        $this->assertEquals(['key' => 'value'], $results[0]->metadata);
    }

    public function test_add_documents(): void
    {
        $sentences = require __DIR__ . '/../stubs/sentences.php';

        $store = new PgVectorStore($this->connection, $this->table, 'embedding_index', count($sentences[0]['embedding']));

        $documents = [];
        foreach ($sentences as $sentence) {
            $document = new Document($sentence['content']);
            $document->embedding = $sentence['embedding'];
            $document->metadata = $sentence['metadata'] ?? [];
            $documents[] = $document;
        }

        $store->addDocuments($documents);

        $results = $store->similaritySearch([0.1, 0.2, 0.3, 0.4, 0.8], 2);
        $this->assertIsIterable($results);
        $this->assertNotEmpty($results);
        $this->assertCount(2, $results);
    }

    public function test_use_several_indexes(): void
    {
        $store1 = new PgVectorStore($this->connection, $this->table, 'embedding_index_1', count($this->embedding));
        $store2 = new PgVectorStore($this->connection, $this->table, 'embedding_index_2', count($this->embedding));

        $document1 = new Document('Hello World!');
        $document1->embedding = $this->embedding;

        $document2 = new Document('Goodbye World!');
        $document2->embedding = $this->embedding;

        $store1->addDocument($document1);
        $store2->addDocument($document2);

        $results1 = $store1->similaritySearch($this->embedding);
        $results2 = $store2->similaritySearch($this->embedding);

        $this->assertIsIterable($results1);
        $this->assertNotEmpty($results1);
        $this->assertCount(1, $results1);
        $this->assertInstanceOf(Document::class, $results1[0]);
        $this->assertEquals('Hello World!', $results1[0]->content);

        $this->assertIsIterable($results2);
        $this->assertNotEmpty($results2);
        $this->assertCount(1, $results2);
        $this->assertInstanceOf(Document::class, $results2[0]);
        $this->assertEquals('Goodbye World!', $results2[0]->content);
    }
}
