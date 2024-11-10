"""
This file contains functions for converting the wiki into a format
that the LLM agent understands better.
"""

import re


def parse_crafting_recipe(match):
    recipe = {}
    # Split the match string by top-level '|' characters,
    # ignoring those within double square brackets
    lines = re.split(r"\|(?![^\[]*\]\])", match)
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            recipe[key.strip()] = value.strip()

    return recipe


# TODO: this doesn't work in some cases.
# - Inventory crafting recipes
# - The bed crafting recipe
def get_ingredient_quantities(recipe):
    slots = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    ingredient_counts = {}
    corresponding_output = {}
    outputs = recipe["Output"].split(";")
    outputs = [output.strip() for output in outputs]

    for slot in slots:
        if slot not in recipe:
            continue

        ingredients = recipe[slot].split(";")
        ingredients = [ingredient.strip() for ingredient in ingredients]

        if len(ingredients) == len(outputs):
            for output, ingredient in zip(outputs, ingredients):
                if not ingredient:
                    continue

                if ingredient not in ingredient_counts:
                    ingredient_counts[ingredient] = 0
                    corresponding_output[ingredient] = output
                ingredient_counts[ingredient] += 1
        else:
            for ingredient in ingredients:
                if not ingredient:
                    continue

                if ingredient not in ingredient_counts:
                    ingredient_counts[ingredient] = 0
                ingredient_counts[ingredient] += 1

    # Group ingredients by their counts
    grouped_ingredients = {}
    for ingredient, count in ingredient_counts.items():
        if count not in grouped_ingredients:
            grouped_ingredients[count] = []

        if ingredient in corresponding_output:
            grouped_ingredients[count].append(
                [ingredient, count, corresponding_output[ingredient]]
            )
        else:
            grouped_ingredients[count].append([ingredient, count])

    # Convert the resulting format to a list of lists
    result = []
    for count in sorted(grouped_ingredients.keys(), reverse=True):
        result.append(grouped_ingredients[count])

    return result


def format_ingredients(grouped_ingredients):
    formatted_ingredients = []
    for group in grouped_ingredients:
        if len(group) > 1:
            formatted_ingredients.append("* One of the following:")
            for ingredient in group:
                if len(ingredient) == 3:
                    formatted_ingredients.append(
                        f"  - {ingredient[1]} {ingredient[0]} (used to craft {ingredient[2]})"
                    )
                else:
                    formatted_ingredients.append(f"  - {ingredient[1]} {ingredient[0]}")
        else:
            formatted_ingredients.append(f"* {group[0][1]} {group[0][0]}")
    return formatted_ingredients


def format_recipe(recipe):
    # name = recipe.get("name", "Unknown Recipe").strip("[]")
    output = recipe.get("Output", "Unknown Output")
    if len(output.split(",")) == 2:
        output_name, output_count = output.split(",")
    else:
        output_name, output_count = output, ""
    description = recipe.get("description", "")

    grouped_ingredients = get_ingredient_quantities(recipe)
    formatted_ingredients = format_ingredients(grouped_ingredients)

    if output_count:
        formatted_recipe = f"Crafting recipe for {output_count} {output_name}:\n"
    else:
        formatted_recipe = f"Crafting recipe for {output_name}:\n"

    if description:
        formatted_recipe += f"Description: {description}\n"
    formatted_recipe += "Ingredients (name, quantity):\n"
    formatted_recipe += "\n".join(formatted_ingredients)

    return formatted_recipe


def parse_recipies(text):
    recipies = []
    pattern = re.compile(r"\{\{[Cc]rafting(.*?)\}\}", re.DOTALL)
    matches = pattern.findall(text)

    for match in matches:
        recipe_dict = parse_crafting_recipe(match)
        recipe = format_recipe(recipe_dict)
        print(recipe)
        print()

    return recipies


def replace_crafting_recipes(input_file, output_file):
    assert input_file != output_file

    with open(input_file, "r") as f:
        text = f.read()

    pattern = re.compile(r"\{\{[Cc]rafting(.*?)\}\}", re.DOTALL)
    matches = pattern.findall(text)

    for match in matches:
        recipe_dict = parse_crafting_recipe(match)
        formatted_recipe = format_recipe(recipe_dict)
        text = text.replace(f"{{{{Crafting{match}}}}}", formatted_recipe)

    with open(output_file, "w") as f:
        f.write(text)


def main():
    crafting_text = ""
    with open("wiki/recipies.txt", "r") as f:
        crafting_text = f.read()

    _ = parse_recipies(crafting_text)


if __name__ == "__main__":
    # main()
    replace_crafting_recipes("wiki.txt", "wiki_formatted.txt")
