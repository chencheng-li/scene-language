"""
Here's a program to create a Greek temple using the provided `helper.py`:

"""
from helper import *

"""
a Greek temple
"""

@register()
def greek_temple() -> Shape:
    return concat_shapes(
        library_call("temple_base"),
        library_call("temple_columns"),
        library_call("temple_roof"),
        library_call("temple_pediment"),
        library_call("temple_steps")
    )

@register()
def temple_base() -> Shape:
    return primitive_call(
        "set_cuboid",
        block_type="minecraft:smooth_stone",
        scale=(20, 2, 30),
        fill=True
    )

@register()
def temple_columns() -> Shape:
    def create_column(i):
        return transform_shape(
            library_call("single_column"),
            translation_matrix([i * 4, 2, 0])
        )

    front_columns = loop(5, create_column)
    back_columns = transform_shape(front_columns, translation_matrix([0, 0, 26]))

    def create_side_column(i):
        return transform_shape(
            library_call("single_column"),
            translation_matrix([0, 2, i * 4 + 4])
        )

    left_columns = loop(6, create_side_column)
    right_columns = transform_shape(left_columns, translation_matrix([16, 0, 0]))

    return concat_shapes(front_columns, back_columns, left_columns, right_columns)

@register()
def single_column() -> Shape:
    column_shaft = primitive_call(
        "set_cuboid",
        block_type="minecraft:quartz_pillar",
        block_kwargs={"axis": "y"},
        scale=(2, 10, 2),
        fill=True
    )

    column_base = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:chiseled_quartz_block",
            scale=(3, 1, 3),
            fill=True
        ),
        translation_matrix([-0.5, 0, -0.5])
    )

    column_capital = transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:chiseled_quartz_block",
            scale=(3, 1, 3),
            fill=True
        ),
        translation_matrix([-0.5, 10, -0.5])
    )

    return concat_shapes(column_shaft, column_base, column_capital)

@register()
def temple_roof() -> Shape:
    return transform_shape(
        primitive_call(
            "set_cuboid",
            block_type="minecraft:smooth_quartz",
            scale=(22, 2, 32),
            fill=True
        ),
        translation_matrix([-1, 12, -1])
    )

@register()
def temple_pediment() -> Shape:
    def create_pediment_layer(i):
        width = 22 - i * 2
        return transform_shape(
            primitive_call(
                "set_cuboid",
                block_type="minecraft:smooth_quartz",
                scale=(width, 1, 4),
                fill=True
            ),
            translation_matrix([i, 14 + i, -1])
        )

    front_pediment = loop(5, create_pediment_layer)
    back_pediment = transform_shape(front_pediment, translation_matrix([0, 0, 28]))

    return concat_shapes(front_pediment, back_pediment)

@register()
def temple_steps() -> Shape:
    def create_step(i):
        return primitive_call(
            "set_cuboid",
            block_type="minecraft:smooth_stone_slab",
            block_kwargs={"type": "bottom", "waterlogged": "false"},
            scale=(22, 1, 1),
            fill=True
        )

    steps = loop(3, lambda i: transform_shape(create_step(i), translation_matrix([-1, i, -i - 1])))

    return steps

"""

This program creates a Greek temple with the following features:

1. A solid stone base
2. Columns arranged in a rectangular pattern (5 columns in front and back, 6 on each side)
3. A roof made of smooth quartz
4. Pediments on the front and back
5. Steps leading up to the temple

The temple is designed to be symmetrical and follows classical Greek architectural principles. The use of quartz blocks gives it a marble-like appearance, which is typical of ancient Greek temples.
"""