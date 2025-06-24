<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Stubs;

use NeuronAI\StructuredOutput\Validation\Rules\ArrayOf;
use NeuronAI\StructuredOutput\Validation\Rules\NotBlank;

class Person
{
    #[NotBlank]
    public string $firstName;
    public string $lastName;

    public Address $address;

    /**
     * @var \NeuronAI\Tests\Stubs\Tag[]
     */
    #[ArrayOf(Tag::class, allowEmpty: true)]
    public array $tags;
}
