<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Traits;

trait CheckOpenPort
{
    private function isPortOpen(string $host, int $port, int $timeout = 1): bool
    {
        $connection = @fsockopen($host, $port, $errno, $errstr, $timeout);
        if (is_resource($connection)) {
            fclose($connection);
            return true;
        }
        return false;
    }
}
