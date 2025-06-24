<?php

declare(strict_types=1);

namespace NeuronAI\Tests;

use NeuronAI\RAG\DataLoader\StringDataLoader;
use PHPUnit\Framework\TestCase;

class DataLoaderTest extends TestCase
{
    public function test_string_data_loader(): void
    {
        $documents = StringDataLoader::for('test')->getDocuments();
        $this->assertCount(1, $documents);
        $this->assertEquals('test', $documents[0]->getContent());
    }
}
