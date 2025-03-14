class SequenceAdapter:
    """Base class only for type hinting purposes. Used for Callable[[int, int] SequenceAdapter] types."""

    pass


class TokenAdapter:
    """Base class only for type hinting purposes. Used with Callable[[int, int] TokenAdapter] types."""

    pass


class ConditionalGenerationAdapter:
    """Base class only for type hinting purposes. Used for Callable[[int, int, int, nn.Module] ConditionalGenerationAdapter] types."""

    pass
