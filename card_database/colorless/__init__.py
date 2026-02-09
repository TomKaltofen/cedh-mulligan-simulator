"""Colorless cards shared across commander registries."""

from cedh_mulligan_simulator.card_registry import (
    Card,
    Mana,
    any_color,
    artifact_0,
    artifact_1,
    artifact_creature_0,
    equipment_1,
)

# 0
MEMNITE = artifact_creature_0("memnite")
ORNITHOPTER = artifact_creature_0("ornithopter")
PHYREXIAN_WALKER = artifact_creature_0("phyrexian_walker")
SHIELD_SPHERE = artifact_creature_0("shield_sphere")
WALKING_BALLISTA = artifact_creature_0("walking_ballista")  # X cost, treating as 0
LOTUS_PETAL = artifact_0("lotus_petal", produces=any_color(1), sacrifices_on_use=True)
MOX_DIAMOND = artifact_0("mox_diamond", produces=any_color(1), requires="spare_land")
CHROME_MOX = artifact_0("chrome_mox", produces=any_color(1), requires="pitchable_card")
MOX_OPAL = artifact_0("mox_opal", produces=any_color(1), requires="metalcraft")
MOX_AMBER = artifact_0("mox_amber")
JEWELED_LOTUS = artifact_0("jeweled_lotus", produces=any_color(3), sacrifices_on_use=True)
LIONS_EYE_DIAMOND = artifact_0(
    "lions_eye_diamond", produces=any_color(3), requires="led_condition", sacrifices_on_use=True
)
JEWELED_AMULET = artifact_0("jeweled_amulet")
TORMODS_CRYPT = artifact_0("tormods_crypt")
GERRARDS_HOURGLASS_PENDANT = artifact_0("gerrards_hourglass_pendant")

# 1
MANA_VAULT = artifact_1("mana_vault", produces=Mana(3))
SOL_RING = artifact_1("sol_ring", produces=Mana(2))
SPRINGLEAF_DRUM = equipment_1("springleaf_drum", tap_produces=any_color(1))
PARADISE_MANTLE = equipment_1("paradise_mantle", tap_produces=any_color(1))
SCROLL_OF_FATE = artifact_1("scroll_of_fate")
VEXING_BAUBLE = artifact_1("vexing_bauble")

# 2
DEFENSE_GRID = Card(name="defense_grid", type="artifact", cost=Mana(2))
DISRUPTOR_FLUTE = Card(name="disruptor_flute", type="artifact", cost=Mana(2))

# 2
GRIM_MONOLITH = Card(name="grim_monolith", type="artifact", cost=Mana(2), produces=Mana(3))
WISHCLAW_TALISMAN = Card(name="wishclaw_talisman", type="artifact", cost=Mana(1))

# 3
STAFF_OF_DOMINATION = Card(name="staff_of_domination", type="artifact", cost=Mana(3))

# 4
THE_ONE_RING = Card(name="the_one_ring", type="artifact", cost=Mana(4))
KARN_THE_GREAT_CREATOR = Card(name="karn_the_great_creator", type="planeswalker", cost=Mana(4))

# Other
CANDELABRA_OF_TAWNOS = Card(name="candelabra_of_tawnos", type="artifact", cost=Mana(1))
EXPEDITION_MAP = Card(name="expedition_map", type="artifact", cost=Mana(1))

# Sideboard - 0 cost
ENGINEERED_EXPLOSIVES = artifact_0("engineered_explosives")  # X cost

# Sideboard - 1 cost
AMULET_OF_VIGOR = Card(name="amulet_of_vigor", type="artifact", cost=Mana(1))
CODEX_SHREDDER = Card(name="codex_shredder", type="artifact", cost=Mana(1))
SOUL_GUIDE_LANTERN = Card(name="soul_guide_lantern", type="artifact", cost=Mana(1))
VOLTAIC_KEY = Card(name="voltaic_key", type="artifact", cost=Mana(1))
CURRENCY_CONVERTER = Card(name="currency_converter", type="artifact", cost=Mana(1))

# Sideboard - 2 cost
GHOST_VACUUM = Card(name="ghost_vacuum", type="artifact", cost=Mana(2))
GLASSES_OF_URZA = Card(name="glasses_of_urza", type="artifact", cost=Mana(1))
MANIFOLD_KEY = Card(name="manifold_key", type="artifact", cost=Mana(2))
SCROLL_RACK = Card(name="scroll_rack", type="artifact", cost=Mana(2))
SEANCE_BOARD = Card(name="seance_board", type="artifact", cost=Mana(2))

# Sideboard - 3 cost
PALANTIR_OF_ORTHANC = Card(name="palantir_of_orthanc", type="artifact", cost=Mana(3))
PRIZED_STATUE = Card(name="prized_statue", type="artifact", cost=Mana(3))

# Sideboard - 4 cost
APPLE_OF_EDEN = Card(name="apple_of_eden", type="artifact", cost=Mana(4))
THRONE_OF_ELDRAINE = Card(name="throne_of_eldraine", type="artifact", cost=Mana(4))

# Sideboard - 5+ cost
KULDOTHA_FORGEMASTER = Card(name="kuldotha_forgemaster", type="creature", cost=Mana(5))
ELDRAZI_CONFLUENCE = Card(name="eldrazi_confluence", type="sorcery", cost=Mana(7))
EMRAKUL_THE_PROMISED_END = Card(name="emrakul_the_promised_end", type="creature", cost=Mana(13))
