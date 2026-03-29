
# ════════════════════════════════════════════════════════════════════
# PyMOL Visualization Script: MYBPC3 A31P Mutation
# Phase 3 - Biomimicry in Healthcare (CRISPR for HCM)
# ════════════════════════════════════════════════════════════════════
# 
# Prerequisites:
#   - Install PyMOL (open-source or educational version)
#   - Download AlphaFold structure: AF-Q14896-F1-model_v4.pdb
#     from https://alphafold.ebi.ac.uk/entry/Q14896
#
# Run: File -> Run Script -> select this file
# ════════════════════════════════════════════════════════════════════

# Reset and configure display
reinitialize
bg_color white
set ray_opaque_background, 1
set ray_shadows, 0

# Fetch the AlphaFold predicted structure for human cMyBP-C
# Q14896 = UniProt ID for human MYBPC3 (cardiac myosin-binding protein C)
fetch AF-Q14896-F1-model_v4, async=0

# Alternative if fetch doesn't work:
# load AF-Q14896-F1-model_v4.pdb

# Rename for clarity
set_name AF-Q14896-F1-model_v4, cMyBPC_WT

# ── FOCUS ON C0 DOMAIN (residues 1-101) ──
# The A31P mutation is in the C0 domain (N-terminal)
create c0_domain, cMyBPC_WT and resi 1-101
hide everything, cMyBPC_WT
show cartoon, c0_domain

# Color the C0 domain by secondary structure
color palegreen, c0_domain and ss H  # helices
color palecyan, c0_domain and ss S   # sheets
color wheat, c0_domain and ss L+''   # loops

# ── HIGHLIGHT POSITION 31 (WILD-TYPE ALA) ──
select wt_ala31, c0_domain and resi 31
show sticks, wt_ala31
color green, wt_ala31
label wt_ala31 and name CA, "Ala31 (WT)"

# Show surrounding residues
select neighbors, c0_domain and (resi 28-34)
show sticks, neighbors
color lightteal, neighbors and not wt_ala31

# ── MODEL THE A31P MUTATION ──
# Create a copy and mutate Ala31 to Pro31
create c0_mutant, c0_domain
# In PyMOL, use the mutagenesis wizard:
# Wizard -> Mutagenesis -> select residue 31 -> Pro -> Apply
# Or use this command:
wizard mutagenesis
# (Manual step: click residue 31, select PRO, apply)

# After mutation, color the mutant residue
# select mut_pro31, c0_mutant and resi 31
# color red, mut_pro31
# label mut_pro31 and name CA, "Pro31 (A31P)"

# ── DISPLAY SETTINGS ──
# Show hydrogen bonds around position 31
select hbond_region, c0_domain and resi 28-34
distance hbonds, hbond_region, hbond_region, 3.5, mode=2

set cartoon_transparency, 0.3, c0_domain
set stick_radius, 0.15
set label_size, 14
set label_color, black
set label_font_id, 7

# Position camera on the C0 domain
orient c0_domain
zoom c0_domain, 5

# ── ANNOTATIONS ──
# Add a title pseudoatom for reference
pseudoatom title_atom, pos=[0, 30, 0], label="MYBPC3 C0 Domain - A31P Mutation Site"
set label_size, 18, title_atom
set label_color, navy, title_atom

# ── RAY TRACE FOR HIGH-QUALITY OUTPUT ──
# Uncomment to generate publication-quality image:
# ray 2400, 1800
# png mybpc3_a31p_structure.png, dpi=300

print("\n" + "=" * 60)
print("MYBPC3 C0 Domain Visualization Loaded")
print("=" * 60)
print("Green sticks: Wild-type Ala31")
print("Use Wizard > Mutagenesis to model Pro31 mutation")
print("The proline's rigid ring disrupts local backbone flexibility")
print("=" * 60)
