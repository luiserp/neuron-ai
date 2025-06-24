<?php

declare(strict_types=1);

namespace NeuronAI\RAG\VectorStore;

use NeuronAI\RAG\Document;

interface VectorStoreInterface
{
    public function addDocument(Document $document): void;

    /**
     * @param  Document[]  $documents
     */
    public function addDocuments(array $documents): void;

    /**
     * Return docs most similar to the embedding.
     *
     * @param  float[]  $embedding
     * @return Document[]
     */
    public function similaritySearch(array $embedding): iterable;
}
