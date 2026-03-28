"""Black cards shared across commander registries."""

from cedh_mulligan_simulator.card_registry import Card, Mana, creature, ritual, sacrifice_outlet, tutor

_B = Mana(1, black=1)
_BB = Mana(2, black=2)
_BBB = Mana(3, black=3)
_1B = Mana(2, black=1)
_2B = Mana(3, black=1)
_2BB = Mana(4, black=2)
_3B = Mana(4, black=1)
_1BBB = Mana(4, black=3)
_4B = Mana(5, black=1)

# Rituals
DARK_RITUAL = ritual("dark_ritual", cost=_B, produces=Mana(3, black=3))
CABAL_RITUAL = ritual("cabal_ritual", cost=_1B, produces=Mana(3, black=3))
CULLING_THE_WEAK = sacrifice_outlet("culling_the_weak", cost=_B, produces_base=Mana(4, black=4))
# Rain of Filth and Songs of the Damned produce variable mana based on runtime board state
# (number of lands / creatures in graveyard). Excluded from mana calculation. They produce
# Mana() here because the static engine cannot model their runtime-dependent yields.
RAIN_OF_FILTH = ritual("rain_of_filth", cost=_B, produces=Mana())
SONGS_OF_THE_DAMNED = ritual("songs_of_the_damned", cost=_B, produces=Mana())
SACRIFICE = sacrifice_outlet("sacrifice", cost=_B, produces_cmc=True)

# Creatures - 0-cost / free
GRIEF = creature("grief", cost=_3B, evoke_cost=Mana(0), cmc=4)  # Evoke: pitch a black card
OFFALSNOUT = creature("offalsnout", evoke_cost=_B, cmc=3)
DEEPWOOD_LEGATE = creature("deepwood_legate", cost=Mana(), cmc=4, free_condition="swamp")
FAERIE_MACABRE = creature("faerie_macabre", cost=Mana(3, black=2), cmc=3)  # Can pitch for free GY hate
STREET_WRAITH = creature("street_wraith", cost=Mana(5, black=1), cmc=5)  # Can cycle for 2 life

# Creatures - paid
BLOODGHAST = creature("bloodghast", cost=_BB, cmc=2)
CYNICAL_LONER = creature("cynical_loner", cost=_2B, cmc=3)
DISCIPLES_OF_GIX = creature("disciples_of_gix", cost=Mana(6, black=2), cmc=6)
EUMIDIAN_HATCHERY = creature("eumidian_hatchery", cost=_2B, cmc=3)
PHYREXIAN_DEVOURER = creature("phyrexian_devourer", cost=Mana(6), cmc=6)
SKIRGE_FAMILIAR = creature("skirge_familiar", cost=_3B, cmc=4)
HOARDING_BROODLORD = creature("hoarding_broodlord", cost=Mana(5, black=1), cmc=6)
OPPOSITION_AGENT = creature("opposition_agent", cost=_2B, cmc=3)
ORCISH_BOWMASTERS = creature("orcish_bowmasters", cost=_1B, cmc=2)
RAZAKETH_THE_FOULBLOODED = creature("razaketh_the_foulblooded", cost=Mana(6, black=2), cmc=8)
TRAZYN_THE_INFINITE = creature("trazyn_the_infinite", cost=Mana(4, black=1), cmc=5)
VILIS_BROKER_OF_BLOOD = creature("vilis_broker_of_blood", cost=Mana(6, black=2), cmc=8)
KRRIK_SON_OF_YAWGMOTH = creature("krrik_son_of_yawgmoth", cost=Mana(7, black=3), cmc=7)  # Can pay life for B
NECROTIC_OOZE = creature("necrotic_ooze", cost=_2BB, cmc=4)
PRIEST_OF_GIX = creature("priest_of_gix", cost=_2B, cmc=3, produces=Mana(3, black=3))

# Top deck Tutors
VAMPIRIC_TUTOR = tutor("vampiric_tutor", cost=_B, destination="top")
IMPERIAL_SEAL = tutor("imperial_seal", cost=_B, destination="top")
SCHEMING_SYMMETRY = tutor("scheming_symmetry", cost=_B, destination="top")
INSIDIOUS_DREAMS = tutor("insidious_dreams", cost=_3B, destination="top")

# Graveyard Tutors
BURIED_ALIVE = tutor("buried_alive", cost=_2B, destination="graveyard")
ENTOMB = tutor("entomb", cost=_B, destination="graveyard")
UNMARKED_GRAVE = tutor("unmarked_grave", cost=_1B, destination="graveyard")

# Hand Tutors
DEMONIC_COUNSEL = tutor("demonic_counsel", cost=_1B, destination="hand")
DEMONIC_TUTOR = tutor("demonic_tutor", cost=_1B, destination="hand")
DIABOLIC_INTENT = tutor("diabolic_intent", cost=_1B, destination="hand")
DIABOLIC_INTENT_2 = tutor("diabolic_intent_2", cost=_1B, destination="hand")
BESEECH_THE_MIRROR = tutor("beseech_the_mirror", cost=_1BBB, destination="battlefield")
MAUSOLEUM_SECRETS = tutor("mausoleum_secrets", cost=_1B, destination="hand")
TAINTED_PACT = tutor("tainted_pact", cost=_1B, destination="hand")
PRAETORS_GRASP = tutor("praetors_grasp", cost=_1BBB, destination="exile")  # From opponent's library

# Reanimation
ANIMATE_DEAD = Card(name="animate_dead", type="reanimation", cost=_1B)
CORPSE_DANCE = Card(name="corpse_dance", type="reanimation", cost=_2B)
DREAD_RETURN = Card(name="dread_return", type="reanimation", cost=_2BB)
GORYOS_VENGEANCE = Card(name="goryos_vengeance", type="reanimation", cost=_1B)
INCARNATION_TECHNIQUE = Card(name="incarnation_technique", type="reanimation", cost=Mana(5, black=3))
NECROMANCY = Card(name="necromancy", type="reanimation", cost=_2B)
REANIMATE = Card(name="reanimate", type="reanimation", cost=_B)
SAW_IN_HALF = Card(name="saw_in_half", type="reanimation", cost=_2B)
SHALLOW_GRAVE = Card(name="shallow_grave", type="reanimation", cost=_1B)
DANCE_OF_THE_DEAD = Card(name="dance_of_the_dead", type="reanimation", cost=_1B)

# Removal
BITTER_TRIUMPH = Card(name="bitter_triumph", type="removal", cost=_1B)
CUT_DOWN = Card(name="cut_down", type="removal", cost=_B)
DEADLY_ROLLICK = Card(name="deadly_rollick", type="removal", cost=Mana())  # Free with commander
DISMEMBER = Card(name="dismember", type="removal", cost=Mana(3))  # Can be paid with phyrexian mana
FATAL_PUSH = Card(name="fatal_push", type="removal", cost=_B)
SICKENING_SHOAL = Card(name="sickening_shoal", type="removal", cost=_BB)  # Can pitch black card
SNUFF_OUT = Card(name="snuff_out", type="removal", cost=Mana(4))  # Can be free
TOXIC_DELUGE = Card(name="toxic_deluge", type="removal", cost=_2B)
CONTAGION = Card(name="contagion", type="removal", cost=Mana(5, black=2))  # Can pitch 2 cards
FLARE_OF_MALICE = Card(name="flare_of_malice", type="removal", cost=_3B)  # Free with sac creature

# Interaction
CABAL_THERAPY = Card(name="cabal_therapy", type="removal", cost=_B)
IMPS_MISCHIEF = Card(name="imps_mischief", type="interaction", cost=_1B)
WORD_OF_COMMAND = Card(name="word_of_command", type="engine", cost=_BB)

# Engines / win-cons
BOLAS_CITADEL = Card(name="bolas_citadel", type="engine", cost=Mana(6))
NECROPOTENCE = Card(name="necropotence", type="engine", cost=_BBB)
PEER_INTO_THE_ABYSS = Card(name="peer_into_the_abyss", type="engine", cost=Mana(7, black=3))
YAWGMOTHS_WILL = Card(name="yawgmoths_will", type="engine", cost=_2B)

# Enchantments
FAITH_OF_THE_DEVOTED = Card(name="faith_of_the_devoted", type="enchantment", cost=_2B)

# Instants
NOT_DEAD_AFTER_ALL = Card(name="not_dead_after_all", type="instant", cost=_B)
AD_NAUSEAM = Card(name="ad_nauseam", type="instant", cost=Mana(5, black=2))
MARCH_OF_WRETCHED_SORROW = Card(name="march_of_wretched_sorrow", type="instant", cost=_B)  # X cost
LONG_GOODBYE = Card(name="long_goodbye", type="removal", cost=_1B)
NOWHERE_TO_RUN = Card(name="nowhere_to_run", type="removal", cost=_2B)
SLAUGHTER_PACT = Card(name="slaughter_pact", type="removal", cost=Mana(0))

# Sorceries
BALEFUL_MASTERY = Card(name="baleful_mastery", type="removal", cost=_3B)  # Can pay 1B
MASSACRE = Card(name="massacre", type="removal", cost=_2BB)  # Can be free vs white
MISINFORMATION = Card(name="misinformation", type="sorcery", cost=_B)
SHRED_MEMORY = Card(name="shred_memory", type="sorcery", cost=_1B)  # Transmute 1BB

# Sideboard Creatures
ANCIENT_CELLARSPAWN = creature("ancient_cellarspawn", cost=Mana(7), cmc=7)
DAUTHI_VOIDWALKER = creature("dauthi_voidwalker", cost=_BB, cmc=2)
EMPEROR_OF_BONES = creature("emperor_of_bones", cost=_1B, cmc=2)
HARVESTER_OF_MISERY = creature("harvester_of_misery", cost=Mana(7, black=2), cmc=7, evoke_cost=Mana(0))  # Pitch black
KAERVEK_THE_PUNISHER = creature("kaervek_the_punisher", cost=_2B, cmc=3)
MESMERIC_FIEND = creature("mesmeric_fiend", cost=_1B, cmc=2)
MYOJIN_OF_NIGHTS_REACH = creature("myojin_of_nights_reach", cost=Mana(8, black=4), cmc=8)
NASHI_MOON_SAGES_SCION = creature("nashi_moon_sages_scion", cost=_2BB, cmc=4)
OVEREAGER_APPRENTICE = creature("overeager_apprentice", cost=_2B, cmc=3, produces=Mana(3, black=3))  # Sac + discard
REV_TITHE_EXTRACTOR = creature("rev_tithe_extractor", cost=_1B, cmc=2)
SOLDEVI_ADNATE = creature("soldevi_adnate", cost=_1B, cmc=2)  # Sac for mana

# Sideboard Tutors
GRIM_TUTOR = tutor("grim_tutor", cost=_1BBB, destination="hand")

# Sideboard Enchantments
BLACK_MARKET_CONNECTIONS = Card(name="black_market_connections", type="enchantment", cost=_2B)
BLOODCHIEF_ASCENSION = Card(name="bloodchief_ascension", type="enchantment", cost=_B)
CASE_OF_THE_STASHED_SKELETON = Card(name="case_of_the_stashed_skeleton", type="enchantment", cost=_1B)
NECRODOMINANCE = Card(name="necrodominance", type="enchantment", cost=_BBB)
TITHING_BLADE = Card(name="tithing_blade", type="artifact", cost=_2B)  # Transforms to creature

# Sideboard Rituals
BUBBLING_MUCK = ritual("bubbling_muck", cost=_B, produces=Mana())  # Variable

# Sideboard Other
COME_BACK_WRONG = Card(name="come_back_wrong", type="reanimation", cost=_1B)
MIRE_IN_MISERY = Card(name="mire_in_misery", type="sorcery", cost=_1B)
VATS = Card(name="vats", type="instant", cost=_2B)
