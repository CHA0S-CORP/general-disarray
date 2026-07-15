"""Unit tests for the DRINK_RECIPE tool (TheCocktailDB, mocked)."""
import pytest

from plugins import drink_tool
from plugins.drink_tool import DrinkRecipeTool, _parse_drink, _spoken_recipe
from tool_plugins import ToolStatus

pytestmark = pytest.mark.unit

MARGARITA = {
    "strDrink": "Margarita",
    "strCategory": "Ordinary Drink",
    "strAlcoholic": "Alcoholic",
    "strGlass": "Cocktail glass",
    "strInstructions": "Rub the rim of the glass with the lime slice.\nShake with ice.",
    "strIngredient1": "Tequila", "strMeasure1": "1 1/2 oz ",
    "strIngredient2": "Triple sec", "strMeasure2": "1/2 oz ",
    "strIngredient3": "Lime juice", "strMeasure3": "1 oz ",
    "strIngredient4": "Salt", "strMeasure4": "",
    "strIngredient5": "", "strMeasure5": None,
}


def _patch(monkeypatch, payload, captured=None):
    async def fake(url, params=None, headers=None):
        if captured is not None:
            captured["url"] = url
            captured["params"] = params
        return payload
    monkeypatch.setattr(drink_tool, "_fetch_json", fake)


def test_parse_drink_collects_ingredients():
    recipe = _parse_drink(MARGARITA)
    assert recipe["name"] == "Margarita"
    assert len(recipe["ingredients"]) == 4
    assert recipe["ingredients"][0] == {"ingredient": "Tequila",
                                        "measure": "1 1/2 oz"}
    assert recipe["ingredients"][3] == {"ingredient": "Salt", "measure": ""}


def test_spoken_recipe_reads_naturally():
    spoken = _spoken_recipe(_parse_drink(MARGARITA))
    assert spoken.startswith("Here's the Margarita.")
    # Measurement abbreviations are expanded for TTS ("oz" -> "ounces"), with
    # singular/plural agreement (see expand_units_for_speech).
    assert "You need 1 1/2 ounces Tequila" in spoken
    assert "1/2 ounce Triple sec" in spoken
    assert " oz" not in spoken
    assert ", and Salt." in spoken
    assert "Serve it in a cocktail glass." in spoken
    assert "\n" not in spoken


async def test_by_name(monkeypatch):
    captured = {}
    _patch(monkeypatch, {"drinks": [MARGARITA]}, captured)
    tool = DrinkRecipeTool(assistant=None)
    result = await tool.execute({"name": "margarita"})

    assert result.status == ToolStatus.SUCCESS
    assert captured["url"].endswith("search.php")
    assert captured["params"] == {"s": "margarita"}
    assert "Margarita" in result.message
    assert result.data["glass"] == "Cocktail glass"


async def test_random_when_no_name(monkeypatch):
    captured = {}
    _patch(monkeypatch, {"drinks": [MARGARITA]}, captured)
    tool = DrinkRecipeTool(assistant=None)
    result = await tool.execute({})

    assert result.status == ToolStatus.SUCCESS
    assert captured["url"].endswith("random.php")


async def test_unknown_drink(monkeypatch):
    _patch(monkeypatch, {"drinks": None})
    tool = DrinkRecipeTool(assistant=None)
    result = await tool.execute({"name": "motor oil"})

    assert result.status == ToolStatus.FAILED
    assert "motor oil" in result.message


async def test_network_failure(monkeypatch):
    async def fake(url, params=None, headers=None):
        raise RuntimeError("down")
    monkeypatch.setattr(drink_tool, "_fetch_json", fake)

    tool = DrinkRecipeTool(assistant=None)
    result = await tool.execute({"name": "margarita"})
    assert result.status == ToolStatus.FAILED
