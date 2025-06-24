<?php

declare(strict_types=1);

namespace NeuronAI\Tests\VectorStore;

use NeuronAI\RAG\Document;
use NeuronAI\RAG\VectorStore\MemoryVectorStore;
use NeuronAI\RAG\VectorStore\VectorStoreInterface;
use PHPUnit\Framework\TestCase;

class MemoryVectorStoreTest extends TestCase
{
    protected array $embedding;

    protected function setUp(): void
    {
        // embedding "Hello World!"
        $this->embedding = json_decode(file_get_contents(__DIR__ . '/../Stubs/hello-world.embeddings'), true);
    }

    public function test_memory_store_instance(): void
    {
        $store = new MemoryVectorStore();
        $this->assertInstanceOf(VectorStoreInterface::class, $store);
    }

    public function test_add_document_and_search(): void
    {
        $this->expectNotToPerformAssertions();
        $document = new Document('Hello World!');
        $document->embedding = $this->embedding;

        $store = new MemoryVectorStore();
        $store->addDocument($document);

        $results = $store->similaritySearch($this->embedding);
    }

    public function test_similarity_search_with_scores(): void
    {
        $vectorStore = new MemoryVectorStore();

        $doc1 = new Document("Document 1");
        $doc1->embedding = [1, 0];
        $doc2 = new Document("Document 2");
        $doc2->embedding = [0, 1];
        $doc3 = new Document("Document 3");
        $doc3->embedding = [0.5, 0.5];

        $vectorStore->addDocuments([$doc1, $doc2, $doc3]);

        $results = $vectorStore->similaritySearch([1, 0]);

        $this->assertCount(3, $results);
        $this->assertGreaterThanOrEqual($results[1]->getScore(), $results[0]->getScore());
        $this->assertGreaterThanOrEqual($results[2]->getScore(), $results[1]->getScore());
    }

    public function test_custom_document_model(): void
    {
        $document = new Document('Hello World!');
        $document->addMetadata('customProperty', 'customValue');
        $document->embedding = [1, 0];

        $vectorStore = new MemoryVectorStore();
        $vectorStore->addDocuments([$document]);

        $results = $vectorStore->similaritySearch([1, 0]);

        $this->assertCount(1, $results);
        $this->assertEquals($document->metadata['customProperty'], $results[0]->metadata['customProperty']);
    }
}
