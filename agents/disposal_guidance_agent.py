"""
Disposal Guidance Agent
-----------------------
Provides proper disposal instructions for each waste category,
including bin colour, hazard warnings, and local authority guidance.
"""

from __future__ import annotations

_DISPOSAL_GUIDANCE: dict[str, dict] = {
    "battery": {
        "bin": "Hazardous Waste / Battery Collection Point",
        "colour_hint": "⚠️ Do NOT put in any household bin.",
        "instructions": (
            "Batteries must be taken to a designated collection point. "
            "Many supermarkets and electronics retailers provide free battery drop-off boxes. "
            "Alternatively, use your local household hazardous waste facility."
        ),
        "hazard_level": "High",
    },
    "biological": {
        "bin": "Green / Organic / Food Waste Bin",
        "colour_hint": "🟢 Green or brown bin depending on your local authority.",
        "instructions": (
            "Place food waste and organic matter in your food/green-waste bin. "
            "If available, home composting is the most sustainable option. "
            "Do not include meat, fish, or cooked food in a home compost bin."
        ),
        "hazard_level": "Low",
    },
    "brown-glass": {
        "bin": "Glass Recycling Bin / Bottle Bank",
        "colour_hint": "🟤 Brown section of the bottle bank.",
        "instructions": (
            "Place rinsed brown glass bottles and jars in the brown glass section "
            "of your local bottle bank, or in your kerbside glass recycling box/bin."
        ),
        "hazard_level": "Low",
    },
    "cardboard": {
        "bin": "Blue / Paper & Cardboard Recycling Bin",
        "colour_hint": "🔵 Blue recycling bin or paper bank.",
        "instructions": (
            "Flatten cardboard boxes and place in the recycling bin. "
            "Remove plastic film, foam, or polystyrene before recycling."
        ),
        "hazard_level": "None",
    },
    "clothes": {
        "bin": "Textile Recycling Bank / Charity Donation",
        "colour_hint": "👕 Textiles should NOT go in general recycling bins.",
        "instructions": (
            "Place wearable items in charity donation bags or clothes banks. "
            "For non-wearable textiles, use a textile recycling bank (often found in supermarket car parks)."
        ),
        "hazard_level": "None",
    },
    "green-glass": {
        "bin": "Glass Recycling Bin / Bottle Bank",
        "colour_hint": "🟢 Green section of the bottle bank.",
        "instructions": (
            "Rinse green glass bottles and jars and place them in the green glass section "
            "of the bottle bank or kerbside glass recycling."
        ),
        "hazard_level": "Low",
    },
    "metal": {
        "bin": "Blue / Recycling Bin",
        "colour_hint": "🔵 Blue recycling bin.",
        "instructions": (
            "Rinse metal cans, tins, and foil trays and place in your recycling bin. "
            "Large metal items (appliances, bikes) should be taken to a household waste recycling centre."
        ),
        "hazard_level": "None",
    },
    "paper": {
        "bin": "Blue / Paper & Cardboard Recycling Bin",
        "colour_hint": "🔵 Blue recycling bin or paper bank.",
        "instructions": (
            "Keep paper dry and place in the recycling bin. "
            "Shredded paper should be bagged before recycling as it can jam sorting machinery."
        ),
        "hazard_level": "None",
    },
    "plastic": {
        "bin": "Blue / Recycling Bin (check local rules)",
        "colour_hint": "🔵 Blue recycling bin – check which plastic types your council accepts.",
        "instructions": (
            "Check the resin code on the base of the item. Most councils accept codes 1 (PET) and 2 (HDPE). "
            "Rinse containers and remove any food residue before placing in recycling."
        ),
        "hazard_level": "Low",
    },
    "shoes": {
        "bin": "Textile Bank / Charity / Take-Back Scheme",
        "colour_hint": "👟 Do not place in general recycling or household waste bins if avoidable.",
        "instructions": (
            "Donate wearable shoes to charity shops or shoe banks. "
            "Worn-out shoes can be taken to brand take-back schemes (e.g. Nike's Reuse-A-Shoe programme)."
        ),
        "hazard_level": "None",
    },
    "trash": {
        "bin": "Black / Grey Residual Waste Bin",
        "colour_hint": "⚫ Black or grey general waste bin.",
        "instructions": (
            "Non-recyclable waste goes in the general (residual) waste bin. "
            "Try to reduce this category by choosing products with recyclable packaging."
        ),
        "hazard_level": "Low",
    },
    "white-glass": {
        "bin": "Glass Recycling Bin / Bottle Bank",
        "colour_hint": "⬜ Clear/white section of the bottle bank.",
        "instructions": (
            "Rinse clear glass bottles and jars and place in the clear/white glass section "
            "of the bottle bank or kerbside glass recycling box."
        ),
        "hazard_level": "Low",
    },
}


class DisposalGuidanceAgent:
    """Returns disposal instructions for a given waste class."""

    def get_guidance(self, waste_class: str) -> dict:
        """
        Parameters
        ----------
        waste_class : str

        Returns
        -------
        dict with keys: waste_class, bin, colour_hint, instructions, hazard_level
        """
        data = _DISPOSAL_GUIDANCE.get(waste_class.lower())
        if data is None:
            return {
                "waste_class": waste_class,
                "bin": "Unknown",
                "colour_hint": "",
                "instructions": "No disposal guidance available for this category.",
                "hazard_level": "Unknown",
            }
        return {"waste_class": waste_class, **data}
