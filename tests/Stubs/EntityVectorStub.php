<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Stubs;

use Doctrine\ORM\Mapping as ORM;
use NeuronAI\RAG\VectorStore\Doctrine\DoctrineEmbeddingEntityBase;

#[ORM\Entity]
#[ORM\Table(name: 'entity_vector_stub')]
class EntityVectorStub extends DoctrineEmbeddingEntityBase
{
}
