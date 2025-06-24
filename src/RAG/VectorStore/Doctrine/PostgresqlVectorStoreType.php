<?php

declare(strict_types=1);

namespace NeuronAI\RAG\VectorStore\Doctrine;

use Doctrine\ORM\EntityManagerInterface;

class PostgresqlVectorStoreType extends SupportedDoctrineVectorStore
{
    public function getVectorAsString(array $vector): string
    {
        return '['.$this->stringListOf($vector).']';
    }

    public function convertToDatabaseValueSQL(string $sqlExpression): string
    {
        return $sqlExpression;
    }

    public function addCustomisationsTo(EntityManagerInterface $entityManager): void
    {
        $entityManager->getConfiguration()->addCustomStringFunction($this->l2DistanceName(), PostgresqlVectorL2OperatorDql::class);
    }

    public function l2DistanceName(): string
    {
        return 'VEC_DISTANCE_EUCLIDEAN';
    }
}
