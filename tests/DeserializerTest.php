<?php

declare(strict_types=1);

namespace NeuronAI\Tests;

use NeuronAI\StructuredOutput\Deserializer\Deserializer;
use NeuronAI\StructuredOutput\Deserializer\DeserializerException;
use NeuronAI\StructuredOutput\SchemaProperty;
use NeuronAI\Tests\Stubs\DummyEnum;
use NeuronAI\Tests\Stubs\IntEnum;
use NeuronAI\Tests\Stubs\Person;
use NeuronAI\Tests\Stubs\Tag;
use NeuronAI\Tests\Stubs\StringEnum;
use PHPUnit\Framework\TestCase;

class DeserializerTest extends TestCase
{
    public function test_person_deserializer(): void
    {
        $json = '{"firstName": "John", "lastName": "Doe"}';

        $obj = Deserializer::fromJson($json, Person::class);

        $this->assertInstanceOf(Person::class, $obj);
        $this->assertEquals('John', $obj->firstName);
        $this->assertEquals('Doe', $obj->lastName);
    }

    public function test_person_with_address(): void
    {
        $json = '{"firstName": "John", "lastName": "Doe", "address": {"city": "Rome"}}';

        $obj = Deserializer::fromJson($json, Person::class);

        $this->assertInstanceOf(Person::class, $obj);
        $this->assertEquals('Rome', $obj->address->city);
    }

    public function test_deserialize_array(): void
    {
        $json = '{"firstName": "John", "lastName": "Doe", "tags": [{"name": "agent"}]}';

        $obj = Deserializer::fromJson($json, Person::class);

        $this->assertInstanceOf(Person::class, $obj);
        $this->assertInstanceOf(Tag::class, $obj->tags[0]);
        $this->assertEquals('agent', $obj->tags[0]->name);
    }

    public function test_deserialize_string_enum(): void
    {
        $class = new class () {
            #[SchemaProperty]
            public StringEnum $number;
        };

        $json = '{"number": "one"}';

        $obj = Deserializer::fromJson($json, $class::class);

        $this->assertInstanceOf($class::class, $obj);
        $this->assertInstanceOf(StringEnum::class, $obj->number);
        $this->assertEquals('one', $obj->number->value);
    }

    public function test_deserialize_int_enum(): void
    {
        $class = new class () {
            #[SchemaProperty]
            public IntEnum $number;
        };

        $json = '{"number": 1}';

        $obj = Deserializer::fromJson($json, $class::class);

        $this->assertInstanceOf($class::class, $obj);
        $this->assertInstanceOf(IntEnum::class, $obj->number);
        $this->assertEquals(1, $obj->number->value);
    }

    public function test_deserialize_invalid_enum(): void
    {
        $class = new class () {
            #[SchemaProperty]
            public DummyEnum $number;
        };

        $json = '{"number": 1}';

        $this->expectException(DeserializerException::class);

        $obj = Deserializer::fromJson($json, $class::class);
    }

    public function test_deserialize_invalid_input(): void
    {
        $class = new class () {
            #[SchemaProperty]
            public StringEnum $number;
        };

        $json = '{"number": "kangaroo"}';

        $this->expectException(DeserializerException::class);

        $obj = Deserializer::fromJson($json, $class::class);
    }

    public function test_deserialize_null_input(): void
    {
        $class = new class () {
            #[SchemaProperty]
            public StringEnum $number;
        };

        $json = '{"number": null}';

        $obj = Deserializer::fromJson($json, $class::class);
        $this->assertInstanceOf($class::class, $obj);
        $this->assertTrue(! isset($obj->number));
    }
    public function test_deserialize_empty_input(): void
    {
        $class = new class () {
            #[SchemaProperty]
            public StringEnum $number;
        };

        $json = '{}';

        $obj = Deserializer::fromJson($json, $class::class);
        $this->assertInstanceOf($class::class, $obj);
        $this->assertTrue(! isset($obj->number));
    }
}
