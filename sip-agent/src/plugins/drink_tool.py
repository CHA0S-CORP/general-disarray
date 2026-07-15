"""
Drink Recipe Tool Plugin
========================
Cocktail and drink recipes from TheCocktailDB's free API (no key required).

Usage in conversation:
User: "How do I make a margarita?"
LLM: [TOOL:DRINK_RECIPE:name=margarita]

User: "Surprise me with a cocktail."
LLM: [TOOL:DRINK_RECIPE]
"""

import logging
from typing import Any, Dict, List, Optional

from plugins.helpers import fetch_json, expand_units_for_speech
from tool_plugins import BaseTool, ToolResult, ToolStatus
from logging_utils import log_event

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.thecocktaildb.com/api/json/v1/1/search.php"
RANDOM_URL = "https://www.thecocktaildb.com/api/json/v1/1/random.php"

_UNAVAILABLE = "The drink recipe service is not available right now."


async def _fetch_json(url: str, params: Optional[Dict[str, Any]] = None,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    """Module-level HTTP helper so tests can monkeypatch it."""
    return await fetch_json(url, params=params, headers=headers)


def _parse_drink(drink: Dict[str, Any]) -> Dict[str, Any]:
    """TheCocktailDB's strIngredient1..15/strMeasure1..15 -> a clean recipe."""
    ingredients: List[Dict[str, str]] = []
    for i in range(1, 16):
        ingredient = (drink.get(f"strIngredient{i}") or "").strip()
        if not ingredient:
            continue
        measure = (drink.get(f"strMeasure{i}") or "").strip()
        ingredients.append({"ingredient": ingredient, "measure": measure})
    return {
        "name": (drink.get("strDrink") or "").strip(),
        "category": (drink.get("strCategory") or "").strip(),
        "alcoholic": (drink.get("strAlcoholic") or "").strip(),
        "glass": (drink.get("strGlass") or "").strip(),
        "instructions": (drink.get("strInstructions") or "").strip(),
        "ingredients": ingredients,
    }


def _spoken_recipe(recipe: Dict[str, Any]) -> str:
    """One TTS-friendly paragraph: ingredients first, then the steps."""
    parts = [f"Here's the {recipe['name']}."]
    if recipe["ingredients"]:
        spoken_items = []
        for item in recipe["ingredients"]:
            if item["measure"]:
                spoken_items.append(f"{item['measure']} {item['ingredient']}".strip())
            else:
                spoken_items.append(item["ingredient"])
        if len(spoken_items) == 1:
            parts.append(f"You need {spoken_items[0]}.")
        else:
            parts.append("You need " + ", ".join(spoken_items[:-1])
                         + f", and {spoken_items[-1]}.")
    if recipe["glass"]:
        parts.append(f"Serve it in a {recipe['glass'].lower()}.")
    if recipe["instructions"]:
        instructions = recipe["instructions"].replace("\r", " ").replace("\n", " ")
        parts.append(instructions if instructions.endswith(".") else instructions + ".")
    # Expand measurement abbreviations ("2 oz" -> "2 ounces") so TTS speaks
    # them naturally instead of "oh zee". Applied to the whole paragraph:
    # measures appear in both the ingredient list and the instructions.
    return expand_units_for_speech(" ".join(parts))


class DrinkRecipeTool(BaseTool):
    """Cocktail/drink recipes from TheCocktailDB."""

    name = "DRINK_RECIPE"
    description = ("Get a COCKTAIL or mixed-drink recipe by name, or a random "
                   "one when no name is given; the result includes ingredients "
                   "with measures and the preparation steps. Drinks only — this "
                   "cannot look up food recipes (use WEB_SEARCH for those)")
    enabled = True
    speak_result = True  # informational: the recipe itself is spoken

    parameters = {
        "name": {
            "type": "string",
            "description": "Drink to look up (e.g. 'margarita'); omit for a "
                           "random cocktail",
            "required": False,
            "default": "",
        },
    }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        name = str(params.get("name") or "").strip()

        try:
            if name:
                payload = await _fetch_json(SEARCH_URL, params={"s": name})
            else:
                payload = await _fetch_json(RANDOM_URL)
        except Exception as e:
            logger.warning(f"Drink recipe fetch failed: {e}")
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        drinks = (payload or {}).get("drinks") or []
        if not drinks:
            return ToolResult(
                status=ToolStatus.FAILED,
                message=f"I couldn't find a drink called {name}." if name
                        else "I couldn't find a drink recipe right now.")

        recipe = _parse_drink(drinks[0])
        if not recipe["name"] or not (recipe["ingredients"] or recipe["instructions"]):
            return ToolResult(status=ToolStatus.FAILED, message=_UNAVAILABLE)

        message = _spoken_recipe(recipe)
        log_event(logger, logging.INFO, f"Drink recipe: {recipe['name']}",
                  event="drink_recipe", drink=recipe["name"])
        return ToolResult(status=ToolStatus.SUCCESS, message=message, data=recipe)
