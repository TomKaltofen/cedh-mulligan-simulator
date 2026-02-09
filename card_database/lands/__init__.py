"""Land cards shared across commander registries."""

from cedh_mulligan_simulator.card_registry import Mana, any_color, land

_B = Mana(1, black=1)

SWAMP = land("swamp", t1_mana=_B, t2_mana=_B)
SWAMP_2 = land("swamp_2", t1_mana=_B, t2_mana=_B)
SNOW_COVERED_SWAMP = land("snow_covered_swamp", t1_mana=_B, t2_mana=_B)
SNOW_COVERED_SWAMP_2 = land("snow_covered_swamp_2", t1_mana=_B, t2_mana=_B)
MULTIVERSAL_PASSAGE = land("multiversal_passage", t1_mana=_B, t2_mana=_B)
PEAT_BOG = land("peat_bog", t1_mana=Mana(), t2_mana=Mana(2, black=2))
LAKE_OF_THE_DEAD = land("lake_of_the_dead", t1_mana=Mana(), t2_mana=Mana(4, black=4), requires="sacrifice_swamp")
GEMSTONE_CAVERNS = land("gemstone_caverns", t1_mana=any_color(1), t2_mana=any_color(1), requires="exile_from_hand")
PHYREXIAN_TOWER = land("phyrexian_tower", t1_mana=Mana(1), t2_mana=Mana(1), sac_mana=Mana(2, black=2))
WASTES = land("wastes", t1_mana=Mana(1), t2_mana=Mana(1))

# Fetch lands — in mono-black, all fetch a Swamp or Snow-Covered Swamp
_SWAMP_TARGETS = ("swamp", "snow_covered_swamp")
BLOODSTAINED_MIRE = land("bloodstained_mire", t1_mana=_B, t2_mana=_B, fetch_targets=_SWAMP_TARGETS)
MARSH_FLATS = land("marsh_flats", t1_mana=_B, t2_mana=_B, fetch_targets=_SWAMP_TARGETS)
POLLUTED_DELTA = land("polluted_delta", t1_mana=_B, t2_mana=_B, fetch_targets=_SWAMP_TARGETS)
PRISMATIC_VISTA = land("prismatic_vista", t1_mana=_B, t2_mana=_B, fetch_targets=_SWAMP_TARGETS)
VERDANT_CATACOMBS = land("verdant_catacombs", t1_mana=_B, t2_mana=_B, fetch_targets=_SWAMP_TARGETS)

# Other lands
ANCIENT_TOMB = land("ancient_tomb", t1_mana=Mana(2), t2_mana=Mana(2))
URBORG_TOMB_OF_YAWGMOTH = land("urborg", t1_mana=_B, t2_mana=_B)
AGADEEMS_AWAKENING = land("agadeems_awakening", t1_mana=_B, t2_mana=_B)  # Pay 3 life to enter untapped
BOGGART_TRAWLER = land("boggart_trawler", t1_mana=_B, t2_mana=_B)  # Pay 3 life to enter untapped
FELL_THE_PROFANE = land("fell_the_profane", t1_mana=_B, t2_mana=_B)  # Pay 3 life to enter untapped
EMERGENCE_ZONE = land("emergence_zone", t1_mana=Mana(1), t2_mana=Mana(1))
BOSEIJU_WHO_SHELTERS_ALL = land("boseiju", t1_mana=Mana(), t2_mana=Mana(1))
URZAS_SAGA = land("urzas_saga", t1_mana=Mana(1), t2_mana=Mana(1))
TALON_GATES_OF_MADARA = land("talon_gates_of_madara", t1_mana=Mana(1), t2_mana=Mana(1))
EUMIDIAN_HATCHERY = land("eumidian_hatchery", t1_mana=_B, t2_mana=_B)
CABAL_PIT = land("cabal_pit", t1_mana=_B, t2_mana=_B)  # Can sac for -2/-2
CITY_OF_TRAITORS = land("city_of_traitors", t1_mana=Mana(2), t2_mana=Mana(2))
CRYPT_OF_AGADEEM = land("crypt_of_agadeem", t1_mana=Mana(), t2_mana=_B)  # Can tap for more with creatures in GY
VAULT_OF_WHISPERS = land("vault_of_whispers", t1_mana=_B, t2_mana=_B)  # Artifact land

# Taplands
TOMB_FORTRESS = land("tomb_fortress", t1_mana=Mana(), t2_mana=_B)
SPYMASTERS_VAULT = land("spymasters_vault", t1_mana=Mana(), t2_mana=_B)

# Sideboard lands
CAVERN_OF_SOULS = land("cavern_of_souls", t1_mana=Mana(1), t2_mana=Mana(1))  # Or any color for creature type
HORIZON_OF_PROGRESS = land("horizon_of_progress", t1_mana=Mana(1), t2_mana=Mana(1))
LOTUS_VALE = land("lotus_vale", t1_mana=Mana(), t2_mana=any_color(3), requires="sacrifice_lands")

TAKENUMA_ABANDONED_MIRE = land("takenuma_abandoned_mire", t1_mana=_B, t2_mana=_B)
URZAS_CAVE = land("urzas_cave", t1_mana=Mana(1), t2_mana=Mana(1), sac_search_turn=3, sac_search_artifact_cmcs=(0, 1))
