<?php

declare(strict_types=1);

namespace NeuronAI\RAG\VectorStore\Search;

use NeuronAI\Exceptions\VectorStoreException;

class SimilaritySearch
{
    public static function cosine(array $vector1, array $vector2): float|int
    {
        if (\count($vector1) !== \count($vector2)) {
            throw new VectorStoreException('Vectors must have the same length to apply cosine similarity.');
        }

        $dotProduct = 0.0;
        $magnitude1 = 0.0;
        $magnitude2 = 0.0;

        foreach ($vector1 as $key => $value) {
            if (isset($vector2[$key])) {
                $dotProduct += $value * $vector2[$key];
            }
            $magnitude1 += $value ** 2;
        }

        foreach ($vector2 as $value) {
            $magnitude2 += $value ** 2;
        }

        if ($magnitude1 === 0.0 || $magnitude2 === 0.0) {
            return 0.0;
        }

        return 1 - $dotProduct / (sqrt($magnitude1) * sqrt($magnitude2));
    }
}
