"""
Recycling Recommendation Agent
--------------------------------
Given a classified waste category, returns actionable recycling and
reuse guidance.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Knowledge base: recycling tips per waste class
# ---------------------------------------------------------------------------
_RECYCLING_TIPS: dict[str, dict] = {
    "battery": {
        "tips": [
            "Take batteries to a dedicated battery recycling point (e.g. supermarkets, electronics stores).",
            "Never dispose of batteries in regular household bins – they contain toxic heavy metals.",
            "Rechargeable batteries can often be returned to the manufacturer.",
        ],
        "reuse_ideas": ["Some AA/AAA batteries can be recharged with a universal charger."],
    },
    "biological": {
        "tips": [
            "Compost food scraps in a home compost bin or municipal green-waste bin.",
            "Fruit and vegetable peelings make excellent compost.",
        ],
        "reuse_ideas": ["Use coffee grounds as a natural garden fertiliser."],
    },
    "brown-glass": {
        "tips": [
            "Rinse containers and place in the glass recycling bin.",
            "Remove metal lids and recycle them separately.",
        ],
        "reuse_ideas": ["Reuse brown glass bottles as storage containers or vases."],
    },
    "cardboard": {
        "tips": [
            "Flatten boxes to save space in the recycling bin.",
            "Remove any plastic tape or foam inserts before recycling.",
        ],
        "reuse_ideas": ["Use cardboard for craft projects, seedling trays, or packing material."],
    },
    "clothes": {
        "tips": [
            "Donate wearable clothes to charity shops or clothing banks.",
            "Many retailers have in-store textile recycling bins.",
        ],
        "reuse_ideas": ["Old t-shirts make great cleaning rags; denim can be repurposed as insulation."],
    },
    "green-glass": {
        "tips": [
            "Rinse and place in the glass recycling bin.",
            "Keep lids separate – they are usually metal and recyclable.",
        ],
        "reuse_ideas": ["Green wine bottles can be repurposed as decorative vases or candle holders."],
    },
    "metal": {
        "tips": [
            "Rinse food cans and place in the metal recycling bin.",
            "Aluminium cans are 100 % recyclable and have high scrap value.",
        ],
        "reuse_ideas": ["Tin cans make great pencil holders, plant pots, or lanterns."],
    },
    "paper": {
        "tips": [
            "Keep paper dry – wet paper cannot be recycled.",
            "Shredded paper can be composted or used as packing material.",
        ],
        "reuse_ideas": ["Use one-sided printed paper for notes; newspapers as gift wrap or drawer liners."],
    },
    "plastic": {
        "tips": [
            "Check the resin code (1-7) on the bottom – codes 1 (PET) and 2 (HDPE) are most widely accepted.",
            "Rinse containers before recycling to avoid contaminating other recyclables.",
        ],
        "reuse_ideas": ["Plastic bottles can become planters, bird feeders, or storage containers."],
    },
    "shoes": {
        "tips": [
            "Donate wearable shoes to charity or shoe banks.",
            "Many sports brands (e.g. Nike, Adidas) run shoe take-back programmes.",
        ],
        "reuse_ideas": ["Worn-out trainers can be shredded into rubber crumb for sports surfaces."],
    },
    "trash": {
        "tips": [
            "General waste should go to your residual (non-recyclable) bin.",
            "Try to minimise general waste by choosing products with recyclable packaging.",
        ],
        "reuse_ideas": [],
    },
    "white-glass": {
        "tips": [
            "Rinse and place in the glass recycling bin.",
            "Clear glass is the most recyclable glass colour.",
        ],
        "reuse_ideas": ["Clear glass jars are perfect for storing dry goods, spices, or homemade preserves."],
    },
}


class RecyclingRecommendationAgent:
    """Returns recycling tips and reuse ideas for a given waste class."""

    def get_recommendations(self, waste_class: str) -> dict:
        """
        Parameters
        ----------
        waste_class : str
            One of the 12 dataset class names (lowercase, hyphenated).

        Returns
        -------
        dict with keys: waste_class, tips (list[str]), reuse_ideas (list[str])
        """
        data = _RECYCLING_TIPS.get(waste_class.lower())
        if data is None:
            return {
                "waste_class": waste_class,
                "tips": ["No specific recycling tips available for this category."],
                "reuse_ideas": [],
            }
        return {"waste_class": waste_class, **data}
