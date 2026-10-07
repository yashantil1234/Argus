"""Build the 120-case Verifier Benchmark Dataset (80 Dev / 40 Test).

Creates:
  - data/verifier_benchmark/dev.json   (80 cases)
  - data/verifier_benchmark/test.json  (40 cases)

Labels:
  - VERIFIED    (Entailment): exactly/semantically supported
  - REJECTED    (Contradiction): negated, contradictory, false numericals
  - UNCERTAIN   (Neutral): scope overreach, causal leap, speculation, missing
"""

import json
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 80 Dev Cases
# ---------------------------------------------------------------------------

DEV_CASES = [
    # --- VERIFIED (27 cases) ---
    {
        "id": "dev_01",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "genomics",
        "claim_text": "Cas9 requires an adjacent PAM sequence for target binding.",
        "quotes": ["Cas9 requires recognition of a PAM sequence adjacent to the target site for target binding."]
    },
    {
        "id": "dev_02",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "genomics",
        "claim_text": "The seed region of the guide RNA is critical for Cas9 cleavage fidelity.",
        "quotes": ["Mismatches within the guide RNA seed region drastically impair the cleavage fidelity of the Cas9 enzyme."]
    },
    {
        "id": "dev_03",
        "label": "verified",
        "phenomenon": "multi_premise_support",
        "domain": "genomics",
        "claim_text": "SpCas9 specifically targets canonical NGG motifs in double-stranded DNA.",
        "quotes": [
            "Engineered variants of Cas nucleases can expand targetable ranges across genomic loci.",
            "In wild-type Streptococcus pyogenes Cas9 (SpCas9), canonical target recognition specifically requires an NGG motif in double-stranded DNA."
        ]
    },
    {
        "id": "dev_04",
        "label": "verified",
        "phenomenon": "joint_synthesis",
        "domain": "genomics",
        "claim_text": "High-fidelity Cas9 variants reduce off-target cleavage while preserving on-target efficacy.",
        "quotes": [
            "Engineered high-fidelity Cas9 variants substantially lower off-target cleavage genome-wide.",
            "These high-fidelity nucleases maintain comparable on-target cleavage efficacy relative to wild-type enzymes."
        ]
    },
    {
        "id": "dev_05",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "biomedicine",
        "claim_text": "Metformin lowers hepatic glucose production in type 2 diabetes.",
        "quotes": ["The primary antidiabetic mechanism of metformin involves the suppression and reduction of hepatic gluconeogenesis."]
    },
    {
        "id": "dev_06",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "biomedicine",
        "claim_text": "Pembrolizumab is a humanized monoclonal antibody targeting programmed cell death protein 1.",
        "quotes": ["Pembrolizumab is an engineered humanized IgG4 monoclonal antibody that binds to the PD-1 receptor."]
    },
    {
        "id": "dev_07",
        "label": "verified",
        "phenomenon": "joint_synthesis",
        "domain": "pharmacology",
        "claim_text": "Statins inhibit HMG-CoA reductase and decrease circulating LDL cholesterol levels.",
        "quotes": [
            "Statins competitively inhibit 3-hydroxy-3-methylglutaryl-coenzyme A (HMG-CoA) reductase, the rate-limiting enzyme in cholesterol synthesis.",
            "This enzymatic inhibition leads to upregulation of LDL receptors and a subsequent marked reduction in circulating serum LDL cholesterol."
        ]
    },
    {
        "id": "dev_08",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "cell_biology",
        "claim_text": "Telomerase elongation counteracts cellular senescence caused by end-replication shortening.",
        "quotes": ["By synthesizing repetitive telomeric DNA, active telomerase offsets the progressive chromosomal shortening that drives replicative senescence."]
    },
    {
        "id": "dev_09",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "genomics",
        "claim_text": "Base editors introduce targeted point mutations without creating double-strand DNA breaks.",
        "quotes": ["Cytosine base editors enable the programmable conversion of C-G to T-A base pairs without generating double-stranded DNA breaks."]
    },
    {
        "id": "dev_10",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "cell_biology",
        "claim_text": "Ubiquitin-tagged proteins are degraded by the 26S proteasome.",
        "quotes": ["Proteins covalently modified with polyubiquitin chains are recognized and subsequently degraded by the 26S proteasome complex."]
    },
    {
        "id": "dev_11",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "pharmacology",
        "claim_text": "Aspirin irreversibly inhibits cyclooxygenase-1 via covalent acetylation.",
        "quotes": ["Acetylsalicylic acid covalently transfers an acetyl group to a serine residue of COX-1, leading to permanent enzymatic inactivation."]
    },
    {
        "id": "dev_12",
        "label": "verified",
        "phenomenon": "joint_synthesis",
        "domain": "genomics",
        "claim_text": "Prime editing combines a reverse transcriptase with a nicking Cas9 to copy genetic edits into the genome.",
        "quotes": [
            "The prime editing architecture employs an engineered Moloney murine leukemia virus reverse transcriptase fused to an H840A Cas9 nickase.",
            "The prime editing guide RNA directly templates the reverse transcription of desired donor sequences into the target genomic locus."
        ]
    },
    {
        "id": "dev_13",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "ai_ml",
        "claim_text": "Self-attention computes dynamic weights based on pairwise token representations.",
        "quotes": ["The self-attention mechanism derives soft alignment scores between every pair of input tokens through scaled dot-product operations."]
    },
    {
        "id": "dev_14",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "ai_ml",
        "claim_text": "Dropout randomly sets a fraction of activations to zero during neural network training.",
        "quotes": ["During each training forward pass, dropout randomly zeroes out hidden unit activations with probability p to prevent co-adaptation."]
    },
    {
        "id": "dev_15",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "cell_biology",
        "claim_text": "Mitochondria produce ATP primarily through oxidative phosphorylation across the inner membrane.",
        "quotes": ["The electrochemical proton gradient across the mitochondrial inner membrane powers ATP synthase to generate the majority of cellular ATP."]
    },
    {
        "id": "dev_16",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "biomedicine",
        "claim_text": "Insulin signaling stimulates GLUT4 translocation to the plasma membrane in skeletal muscle.",
        "quotes": ["Binding of insulin triggers the phosphorylation cascade that mobilizes intracellular GLUT4 glucose transporters to the plasma membrane in myocytes."]
    },
    {
        "id": "dev_17",
        "label": "verified",
        "phenomenon": "joint_synthesis",
        "domain": "pharmacology",
        "claim_text": "Checkpoint inhibitors disinhibit cytotoxic T cells to mount an antitumor immune response.",
        "quotes": [
            "Blockade of CTLA-4 or PD-1 removes key inhibitory brakes on antigen-experienced cytotoxic T lymphocytes.",
            "This functional reactivation unleashes tumor-infiltrating lymphocytes to destroy malignant neoplastic cells."
        ]
    },
    {
        "id": "dev_18",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "genomics",
        "claim_text": "DNA methylation at CpG islands typically suppresses downstream gene transcription.",
        "quotes": ["Hypermethylation of cytosine residues within promoter CpG islands recruits transcriptional repressors, causing stable gene silencing."]
    },
    {
        "id": "dev_19",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "biomedicine",
        "claim_text": "C-reactive protein is an acute-phase reactant synthesized by hepatocytes in response to IL-6.",
        "quotes": ["C-reactive protein (CRP) is an acute-phase protein produced primarily by hepatocytes following stimulation by interleukin-6."]
    },
    {
        "id": "dev_20",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "cell_biology",
        "claim_text": "p53 acts as a transcription factor regulating genes involved in cell cycle arrest and apoptosis.",
        "quotes": ["The tumor suppressor p53 functions as a sequence-specific transcription factor that activates expression of downstream effectors mediating cell cycle arrest and apoptotic death."]
    },
    {
        "id": "dev_21",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "genomics",
        "claim_text": "Histone acetylation neutralizes lysine positive charges, promoting an open chromatin structure.",
        "quotes": ["Acetylation by histone acetyltransferases abolishes the positive charge on lysine side chains, loosening histone-DNA interactions and increasing chromatin accessibility."]
    },
    {
        "id": "dev_22",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "ai_ml",
        "claim_text": "Layer normalization standardizes inputs across feature dimensions within each training example.",
        "quotes": ["Unlike batch normalization, layer normalization computes the mean and variance across all features for a single sample independently."]
    },
    {
        "id": "dev_23",
        "label": "verified",
        "phenomenon": "joint_synthesis",
        "domain": "biomedicine",
        "claim_text": "CAR-T cells express synthetic receptors targeting tumor surface antigens to induce cytolysis.",
        "quotes": [
            "Chimeric antigen receptors (CARs) fuse an extracellular single-chain antibody fragment to intracellular T-cell activation domains.",
            "Upon binding specific tumor surface antigens, engineered CAR-T cells release perforin and granzymes, mediating targeted tumor cell lysis."
        ]
    },
    {
        "id": "dev_24",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "pharmacology",
        "claim_text": "Warfarin acts as a vitamin K antagonist, inhibiting the synthesis of functional clotting factors.",
        "quotes": ["Warfarin suppresses vitamin K epoxide reductase, preventing gamma-carboxylation and maturation of clotting factors II, VII, IX, and X."]
    },
    {
        "id": "dev_25",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "genomics",
        "claim_text": "Alternative splicing enables a single pre-mRNA to yield distinct protein isoforms.",
        "quotes": ["Differential exon inclusion during pre-mRNA splicing permits a single genetic locus to generate multiple structurally distinct mature polypeptide variants."]
    },
    {
        "id": "dev_26",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "cell_biology",
        "claim_text": "Autophagy packages cytoplasmic components into double-membrane autophagosomes for lysosomal digestion.",
        "quotes": ["Macroautophagy sequesters damaged organelles within double-membrane vesicles called autophagosomes, which subsequently fuse with lysosomes for degradation."]
    },
    {
        "id": "dev_27",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "biomedicine",
        "claim_text": "GLP-1 receptor agonists enhance glucose-dependent insulin secretion and delay gastric emptying.",
        "quotes": ["Glucagon-like peptide-1 (GLP-1) receptor agonists augment glucose-stimulated insulin release from beta cells while decelerating gastric transit."]
    },

    # --- REJECTED (27 cases) ---
    {
        "id": "dev_28",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "genomics",
        "claim_text": "Calcium ions significantly enhance Cas9 cleavage activity.",
        "quotes": ["Cas9 cleavage activity is completely inhibited and blocked in the presence of calcium ions."]
    },
    {
        "id": "dev_29",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "genomics",
        "claim_text": "The engineered variant retained over 90% of wild-type kinetic efficiency.",
        "quotes": ["Kinetic measurements revealed that the engineered mutant retained less than 15% of wild-type catalytic efficiency."]
    },
    {
        "id": "dev_30",
        "label": "rejected",
        "phenomenon": "universal_contradiction",
        "domain": "genomics",
        "claim_text": "Cas9 never cleaves DNA sequences containing seed-region mismatches.",
        "quotes": ["Genome-wide profiling confirmed that Cas9 readily cleaves certain non-target loci despite harboring single seed-region mismatches."]
    },
    {
        "id": "dev_31",
        "label": "rejected",
        "phenomenon": "reversibility_contradiction",
        "domain": "genomics",
        "claim_text": "The chemical modification permanently inactivates the Cas9 ribonucleoprotein complex.",
        "quotes": ["Upon exposure to visible light, the chemical modification is cleaved, fully restoring ribonucleoprotein activity within minutes."]
    },
    {
        "id": "dev_32",
        "label": "rejected",
        "phenomenon": "directional_negation",
        "domain": "biomedicine",
        "claim_text": "Metformin therapy causes sharp increases in plasma fasting glucose.",
        "quotes": ["Treatment with metformin produced a consistent, statistically significant decrease in fasting blood glucose across all experimental cohorts."]
    },
    {
        "id": "dev_33",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "pharmacology",
        "claim_text": "The patient group taking drug X showed a mortality rate of 85%.",
        "quotes": ["In the intention-to-treat population, overall 30-day mortality was documented at 4.2% in the drug X arm."]
    },
    {
        "id": "dev_34",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "cell_biology",
        "claim_text": "Bcl-2 overexpression accelerates apoptotic cell death in response to radiation.",
        "quotes": ["Elevated expression of Bcl-2 acts as an anti-apoptotic survival factor, robustly protecting cells from radiation-induced apoptosis."]
    },
    {
        "id": "dev_35",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "genomics",
        "claim_text": "Cas12a nucleases require two independent tracrRNA and crRNA molecules for cleavage.",
        "quotes": ["Unlike Cas9, Cas12a (Cpf1) functions as a single crRNA-guided endonuclease that operates completely independently of tracrRNA."]
    },
    {
        "id": "dev_36",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "ai_ml",
        "claim_text": "The convolutional model achieved a top-1 accuracy below 30% on ImageNet.",
        "quotes": ["Our baseline convolutional network attained a top-1 classification accuracy of 78.4% on the ImageNet validation benchmark."]
    },
    {
        "id": "dev_37",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "cell_biology",
        "claim_text": "Telomerase activity is abundant in most healthy adult human somatic cells.",
        "quotes": ["Telomerase reverse transcriptase expression is transcriptionally silenced in the vast majority of normal human adult somatic tissues."]
    },
    {
        "id": "dev_38",
        "label": "rejected",
        "phenomenon": "directional_negation",
        "domain": "biomedicine",
        "claim_text": "Insulin promotes the release of free fatty acids from adipose tissue.",
        "quotes": ["Insulin exerts potent antilipolytic effects, inhibiting hormone-sensitive lipase and suppressing free fatty acid efflux from adipocytes."]
    },
    {
        "id": "dev_39",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "genomics",
        "claim_text": "The human genome encodes over 500,000 protein-coding genes.",
        "quotes": ["High-throughput sequencing and GENCODE manual curation annotate approximately 19,000 to 20,000 protein-coding genes in the human genome."]
    },
    {
        "id": "dev_40",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "pharmacology",
        "claim_text": "Proton pump inhibitors stimulate gastric acid secretion by parietal cells.",
        "quotes": ["Proton pump inhibitors bind covalently to H+/K+-ATPase pumps, directly blocking hydrogen ion secretion into the gastric lumen."]
    },
    {
        "id": "dev_41",
        "label": "rejected",
        "phenomenon": "universal_contradiction",
        "domain": "cell_biology",
        "claim_text": "Mitochondria lack their own genome and depend entirely on nuclear DNA.",
        "quotes": ["Mitochondria maintain an autonomous, circular double-stranded genome (mtDNA) that encodes 13 core respiratory chain polypeptides."]
    },
    {
        "id": "dev_42",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "ai_ml",
        "claim_text": "Residual connections make gradients vanish more rapidly during deep backpropagation.",
        "quotes": ["Residual skip connections provide an unimpeded gradient highway that mitigates the vanishing gradient problem in deep networks."]
    },
    {
        "id": "dev_43",
        "label": "rejected",
        "phenomenon": "directional_negation",
        "domain": "biomedicine",
        "claim_text": "Glucagon acts to decrease blood glucose during fasting states.",
        "quotes": ["Secreted by pancreatic alpha cells in response to hypoglycemia, glucagon stimulates glycogenolysis and gluconeogenesis to elevate systemic blood glucose."]
    },
    {
        "id": "dev_44",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "genomics",
        "claim_text": "DNA polymerase synthesizes nascent DNA strands in the 3' to 5' direction.",
        "quotes": ["All known cellular DNA polymerases synthesize DNA strictly in the 5' to 3' direction by adding nucleotides to the 3'-OH terminus."]
    },
    {
        "id": "dev_45",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "biomedicine",
        "claim_text": "The antibody titer dropped to zero in all vaccinated animals within 24 hours.",
        "quotes": ["High-affinity neutralizing antibody titers remained robustly elevated through 6 months post-immunization across all treated primates."]
    },
    {
        "id": "dev_46",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "cell_biology",
        "claim_text": "Cyclin-dependent kinases (CDKs) are constitutively active without binding cyclins.",
        "quotes": ["Monomeric CDKs are catalytically inactive; kinase activity strictly requires conformational remodeling upon binding regulatory cyclin subunits."]
    },
    {
        "id": "dev_47",
        "label": "rejected",
        "phenomenon": "directional_negation",
        "domain": "pharmacology",
        "claim_text": "Beta-blockers increase heart rate and myocardial contractility.",
        "quotes": ["Beta-1 adrenergic antagonists competitively inhibit catecholamine binding, reducing heart rate and decreasing cardiac inotropy."]
    },
    {
        "id": "dev_48",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "genomics",
        "claim_text": "RNA interference degrades target messenger RNAs exclusively in the mitochondrial matrix.",
        "quotes": ["The canonical RNA-induced silencing complex (RISC) mediates sequence-specific mRNA cleavage within the eukaryotic cytoplasm."]
    },
    {
        "id": "dev_49",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "ai_ml",
        "claim_text": "The vocabulary size of the tokenizer was restricted to only 50 distinct tokens.",
        "quotes": ["The BPE subword tokenizer was constructed with a vocabulary vocabulary size of 32,000 merge tokens."]
    },
    {
        "id": "dev_50",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "biomedicine",
        "claim_text": "Loss of p53 function sensitizes tumor cells to spontaneous apoptosis.",
        "quotes": ["Inactivation of p53 confers apoptosis resistance, allowing cells harboring oncogenic lesions to evade cell death."]
    },
    {
        "id": "dev_51",
        "label": "rejected",
        "phenomenon": "directional_negation",
        "domain": "cell_biology",
        "claim_text": "Active transport moves solutes down their thermodynamic concentration gradient without energy input.",
        "quotes": ["Primary active transport utilizes ATP hydrolysis to drive solute translocation uphill against electrochemical gradients."]
    },
    {
        "id": "dev_52",
        "label": "rejected",
        "phenomenon": "universal_contradiction",
        "domain": "genomics",
        "claim_text": "Introns are fully retained in mature eukaryotic messenger RNAs.",
        "quotes": ["During pre-mRNA processing, spliceosomes precisely excise non-coding intronic sequences and ligate exons to yield mature mRNA."]
    },
    {
        "id": "dev_53",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "pharmacology",
        "claim_text": "The drug's elimination half-life is greater than 30 days.",
        "quotes": ["Pharmacokinetic profiling established an ultra-rapid clearance profile with an elimination half-life of 45 minutes in healthy volunteers."]
    },
    {
        "id": "dev_54",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "biomedicine",
        "claim_text": "Platelet aggregation is completely prevented by high concentrations of thrombin.",
        "quotes": ["Thrombin acts as the most potent physiological agonist of platelet activation, driving rapid aggregation and fibrin clot stabilization."]
    },

    # --- UNCERTAIN (26 cases) ---
    {
        "id": "dev_55",
        "label": "uncertain",
        "phenomenon": "scope_overreach",
        "domain": "biomedicine",
        "claim_text": "The gene therapy successfully reverses liver fibrosis in human clinical patients.",
        "quotes": ["In a murine model of chronic liver injury, the therapeutic vector successfully reversed hepatic fibrosis in mice."]
    },
    {
        "id": "dev_56",
        "label": "uncertain",
        "phenomenon": "causal_overreach",
        "domain": "genomics",
        "claim_text": "Elevated Cas9 expression directly causes increased chromosomal translocations.",
        "quotes": ["High levels of Cas9 expression were positively correlated with higher observed frequencies of chromosomal translocations."]
    },
    {
        "id": "dev_57",
        "label": "uncertain",
        "phenomenon": "speculative_hedging",
        "domain": "genomics",
        "claim_text": "The loop region determines the binding affinity for modified tracrRNA.",
        "quotes": ["We speculate that this flexible loop region may participate in governing binding affinity for modified tracrRNA."]
    },
    {
        "id": "dev_58",
        "label": "uncertain",
        "phenomenon": "quantitative_overreach",
        "domain": "cell_biology",
        "claim_text": "The treatment increased cell viability by exactly 45 percent.",
        "quotes": ["Treated cohorts exhibited a statistically significant and noticeable increase in overall cell viability."]
    },
    {
        "id": "dev_59",
        "label": "uncertain",
        "phenomenon": "partial_support",
        "domain": "genomics",
        "claim_text": "Cas9 requires ATP hydrolysis and operates optimally at physiological pH.",
        "quotes": ["Biochemical assays demonstrated that Cas9 functions optimally at physiological pH 7.4."]
    },
    {
        "id": "dev_60",
        "label": "uncertain",
        "phenomenon": "irrelevant_evidence",
        "domain": "genomics",
        "claim_text": "Cas9 undergoes conformational change upon guide RNA loading.",
        "quotes": ["Photosynthetic efficiency in Arabidopsis thaliana is regulated by non-photochemical quenching under excess irradiance."]
    },
    {
        "id": "dev_61",
        "label": "uncertain",
        "phenomenon": "outside_knowledge_trap",
        "domain": "genomics",
        "claim_text": "The human genome contains approximately 3 billion base pairs.",
        "quotes": ["Streptococcus pyogenes is a Gram-positive pathogen that harbors a Type II CRISPR-Cas adaptive immune system."]
    },
    {
        "id": "dev_62",
        "label": "uncertain",
        "phenomenon": "missing_evidence",
        "domain": "genomics",
        "claim_text": "CRISPR interference selectively silences gene transcription without cutting DNA.",
        "quotes": []
    },
    {
        "id": "dev_63",
        "label": "uncertain",
        "phenomenon": "scope_overreach",
        "domain": "pharmacology",
        "claim_text": "Compound B-12 cures pancreatic adenocarcinoma in pediatric oncology patients.",
        "quotes": ["Compound B-12 demonstrated nanomolar cytotoxicity against PANC-1 human pancreatic cancer cell lines in 2D cell cultures."]
    },
    {
        "id": "dev_64",
        "label": "uncertain",
        "phenomenon": "causal_overreach",
        "domain": "biomedicine",
        "claim_text": "Dietary sodium restriction directly prevents myocardial infarction.",
        "quotes": ["Epidemiological registries indicate lower dietary sodium intake is associated with reduced incidence of cardiovascular events."]
    },
    {
        "id": "dev_65",
        "label": "uncertain",
        "phenomenon": "speculative_hedging",
        "domain": "cell_biology",
        "claim_text": "Phosphorylation at Ser473 activates the kinase for nuclear translocation.",
        "quotes": ["It is tempting to hypothesize that phosphorylation at Ser473 might conceivably facilitate nuclear translocation of the complex."]
    },
    {
        "id": "dev_66",
        "label": "uncertain",
        "phenomenon": "quantitative_overreach",
        "domain": "ai_ml",
        "claim_text": "The quantized model operates with a latency reduction of exactly 70%.",
        "quotes": ["INT8 quantization yielded substantial latency improvements across mobile edge test devices compared to float32."]
    },
    {
        "id": "dev_67",
        "label": "uncertain",
        "phenomenon": "partial_support",
        "domain": "biomedicine",
        "claim_text": "The therapy eradicates primary tumors and completely blocks metastatic dissemination.",
        "quotes": ["Intratumoral injection of the vector caused complete regression of primary xenograft tumor mass."]
    },
    {
        "id": "dev_68",
        "label": "uncertain",
        "phenomenon": "irrelevant_evidence",
        "domain": "biomedicine",
        "claim_text": "BRCA1 mutations predispose patients to early-onset breast cancer.",
        "quotes": ["Seasonal precipitation shifts in sub-Saharan Africa have altered migratory patterns of locust swarms."]
    },
    {
        "id": "dev_69",
        "label": "uncertain",
        "phenomenon": "outside_knowledge_trap",
        "domain": "biomedicine",
        "claim_text": "Water freezes at zero degrees Celsius under atmospheric pressure.",
        "quotes": ["The crystal structure of bovine serum albumin was solved at 2.1 angstrom resolution using X-ray crystallography."]
    },
    {
        "id": "dev_70",
        "label": "uncertain",
        "phenomenon": "missing_evidence",
        "domain": "pharmacology",
        "claim_text": "Liposomal encapsulation protects small interfering RNA from systemic nuclease degradation.",
        "quotes": []
    },
    {
        "id": "dev_71",
        "label": "uncertain",
        "phenomenon": "scope_overreach",
        "domain": "genomics",
        "claim_text": "The prime editor achieves 80% editing efficiency in all human tissues in vivo.",
        "quotes": ["In HEK293T cells in vitro, prime editing attained targeted installation efficiencies reaching 80% at the test locus."]
    },
    {
        "id": "dev_72",
        "label": "uncertain",
        "phenomenon": "causal_overreach",
        "domain": "biomedicine",
        "claim_text": "Poor sleep quality directly initiates amyloid beta plaque deposition.",
        "quotes": ["Chronic sleep fragmentation was correlated with higher CSF concentrations of amyloid beta peptides in cross-sectional cohorts."]
    },
    {
        "id": "dev_73",
        "label": "uncertain",
        "phenomenon": "speculative_hedging",
        "domain": "pharmacology",
        "claim_text": "The allosteric binding pocket confers absolute resistance to kinase inhibitors.",
        "quotes": ["Computational docking simulations suggest this cryptic allosteric pocket could potentially impair competitor binding."]
    },
    {
        "id": "dev_74",
        "label": "uncertain",
        "phenomenon": "quantitative_overreach",
        "domain": "cell_biology",
        "claim_text": "The longevity intervention doubles average organismal lifespan by exactly 100%.",
        "quotes": ["Nematodes treated with the small molecule exhibited a statistically significant prolongation of median survival."]
    },
    {
        "id": "dev_75",
        "label": "uncertain",
        "phenomenon": "partial_support",
        "domain": "pharmacology",
        "claim_text": "The vaccine confers sterile immunity against transmission and prevents mild symptomatic illness.",
        "quotes": ["Phase 3 trial endpoints demonstrated 94% efficacy in preventing severe clinical disease and hospitalization."]
    },
    {
        "id": "dev_76",
        "label": "uncertain",
        "phenomenon": "irrelevant_evidence",
        "domain": "ai_ml",
        "claim_text": "Transformer attention heads exhibit sparsity across deeper layers.",
        "quotes": ["Geothermal gradients in volcanic rift zones produce geothermal fluids with enriched mineral content."]
    },
    {
        "id": "dev_77",
        "label": "uncertain",
        "phenomenon": "outside_knowledge_trap",
        "domain": "pharmacology",
        "claim_text": "Penicillin was discovered by Alexander Fleming in 1928.",
        "quotes": ["Synthetic antimicrobial peptides display broad-spectrum activity against multidrug-resistant Pseudomonas strains."]
    },
    {
        "id": "dev_78",
        "label": "uncertain",
        "phenomenon": "missing_evidence",
        "domain": "ai_ml",
        "claim_text": "Direct preference optimization eliminates the need for an explicit reward model in RLHF.",
        "quotes": []
    },
    {
        "id": "dev_79",
        "label": "uncertain",
        "phenomenon": "scope_overreach",
        "domain": "cell_biology",
        "claim_text": "The anti-inflammatory cytokine prevents arthritis in geriatric patient populations.",
        "quotes": ["Recombinant cytokine administration reduced joint swelling and cartilage erosion in collagen-induced arthritis rat models."]
    },
    {
        "id": "dev_80",
        "label": "uncertain",
        "phenomenon": "causal_overreach",
        "domain": "biomedicine",
        "claim_text": "Microbiome dysbiosis causes childhood asthma development.",
        "quotes": ["Alterations in early-infancy gut bacterial composition were associated with an elevated risk of subsequent asthma diagnosis."]
    }
]


# ---------------------------------------------------------------------------
# 40 Test Cases (Held-out Test Split — Frozen)
# ---------------------------------------------------------------------------

TEST_CASES = [
    # --- VERIFIED (14 cases) ---
    {
        "id": "test_01",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "genomics",
        "claim_text": "Cas9 cleavage generates predominantly blunt-ended double-stranded breaks.",
        "quotes": ["Target DNA cleavage by wild-type Cas9 produces predominantly blunt double-stranded DNA ends 3 base pairs upstream of the PAM."]
    },
    {
        "id": "dev_test_02",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "genomics",
        "claim_text": "Engineered deaminases modify target nucleotides without double-strand break intermediates.",
        "quotes": ["Base editing complexes catalyze direct deamination of target cytosines or adenines without double-stranded DNA cleavage."]
    },
    {
        "id": "test_03",
        "label": "verified",
        "phenomenon": "joint_synthesis",
        "domain": "pharmacology",
        "claim_text": "Doxorubicin intercalates DNA and inhibits topoisomerase II to block tumor replication.",
        "quotes": [
            "Doxorubicin binds avidly to double-stranded DNA via planar anthracycline ring intercalation.",
            "This interaction stabilizes the topoisomerase II cleavable complex, halting replication forks and precipitating lethal DNA breaks."
        ]
    },
    {
        "id": "test_04",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "cell_biology",
        "claim_text": "Caspase-3 acts as an executioner caspase in mammalian apoptosis.",
        "quotes": ["Cleavage of downstream cellular structural substrates by active caspase-3 constitutes the central execution phase of apoptosis."]
    },
    {
        "id": "test_05",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "biomedicine",
        "claim_text": "HER2 amplification drives aggressive oncogenic growth in a subset of breast cancers.",
        "quotes": ["Amplification of the ERBB2 gene, encoding HER2, occurs in approximately 20% of breast tumors, driving aggressive neoplastic proliferation."]
    },
    {
        "id": "test_06",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "ai_ml",
        "claim_text": "Batch normalization reduces internal covariate shift during deep neural network training.",
        "quotes": ["By standardizing layer inputs across mini-batch examples, batch normalization dramatically accelerates training and mitigates internal covariate shift."]
    },
    {
        "id": "test_07",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "pharmacology",
        "claim_text": "Methotrexate inhibits dihydrofolate reductase to deplete thymidine pools.",
        "quotes": ["Methotrexate acts as a competitive antagonist of dihydrofolate reductase (DHFR), shutting down de novo synthesis of thymidylate and purines."]
    },
    {
        "id": "test_08",
        "label": "verified",
        "phenomenon": "joint_synthesis",
        "domain": "cell_biology",
        "claim_text": "Microtubule dynamic instability is powered by GTP hydrolysis on beta-tubulin subunits.",
        "quotes": [
            "Tubulin heterodimers polymerize into protofilaments in a GTP-bound conformation.",
            "Hydrolysis of GTP bound to beta-tubulin weakens the lattice structure, inducing catastrophe and rapid microtubule depolymerization."
        ]
    },
    {
        "id": "test_09",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "genomics",
        "claim_text": "Okazaki fragments are synthesized discontinuously on the lagging strand during DNA replication.",
        "quotes": ["Because polymerases synthesize exclusively 5' to 3', lagging strand synthesis proceeds discontinuously via short Okazaki segments."]
    },
    {
        "id": "test_10",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "biomedicine",
        "claim_text": "ACE inhibitors reduce blood pressure by suppressing angiotensin II synthesis.",
        "quotes": ["Angiotensin-converting enzyme inhibitors block the conversion of angiotensin I to the potent vasoconstrictor angiotensin II, lowering systemic vascular resistance."]
    },
    {
        "id": "test_11",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "ai_ml",
        "claim_text": "Backpropagation computes analytical gradients using the chain rule of calculus.",
        "quotes": ["The backpropagation algorithm systematically applies the chain rule to compute partial derivatives of loss with respect to all network weights."]
    },
    {
        "id": "test_12",
        "label": "verified",
        "phenomenon": "exact_support",
        "domain": "cell_biology",
        "claim_text": "Gap junctions allow direct intercellular passage of small ions and signaling molecules.",
        "quotes": ["Connexon hexamers dock between adjacent cell membranes to form gap junction channels permeable to ions and metabolites under 1 kDa."]
    },
    {
        "id": "test_13",
        "label": "verified",
        "phenomenon": "joint_synthesis",
        "domain": "pharmacology",
        "claim_text": "Naloxone displaces opioids from mu-opioid receptors to reverse respiratory depression.",
        "quotes": [
            "Naloxone is a pure competitive antagonist with high binding affinity for mu-opioid receptors.",
            "Administration rapidly displaces bound opioid agonists, promptly restoring physiological respiratory drive during acute overdose."
        ]
    },
    {
        "id": "test_14",
        "label": "verified",
        "phenomenon": "semantic_paraphrase",
        "domain": "genomics",
        "claim_text": "CpG island methylation in tumor suppressor promoters silences expression in cancer cells.",
        "quotes": ["Aberrant de novo methylation of promoter CpG islands serves as an epigenetic mechanism that permanently represses tumor suppressor transcription in neoplasia."]
    },

    # --- REJECTED (13 cases) ---
    {
        "id": "test_15",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "genomics",
        "claim_text": "Cas9 cleaves target DNA without any requirement for a guide RNA.",
        "quotes": ["Apo-Cas9 devoid of guide RNA is completely inert and lacks any measurable affinity or cleavage activity on double-stranded DNA."]
    },
    {
        "id": "test_16",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "pharmacology",
        "claim_text": "The antibiotic eradicated bacterial colonies in less than 2 minutes.",
        "quotes": ["Time-kill kinetics demonstrated that bacterial eradication required a minimum of 24 to 48 hours of continuous exposure at 4x MIC."]
    },
    {
        "id": "test_17",
        "label": "rejected",
        "phenomenon": "directional_negation",
        "domain": "biomedicine",
        "claim_text": "Epinephrine administration causes systemic vasodilation and severe hypotension.",
        "quotes": ["Through alpha-1 and beta-1 adrenergic stimulation, epinephrine increases cardiac output and systemic vascular resistance, elevating blood pressure."]
    },
    {
        "id": "test_18",
        "label": "rejected",
        "phenomenon": "universal_contradiction",
        "domain": "cell_biology",
        "claim_text": "Prokaryotic cells contain a membrane-enclosed nucleus and endoplasmic reticulum.",
        "quotes": ["Prokaryotes lack membrane-delimited intracellular organelles; their genetic material resides in an unenclosed cytoplasmic nucleoid region."]
    },
    {
        "id": "test_19",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "ai_ml",
        "claim_text": "ReLU activations produce smooth, continuously differentiable outputs with zero gradient saturation.",
        "quotes": ["The rectified linear unit f(x) = max(0, x) has a non-differentiable sharp corner at x = 0 and outputs a hard zero gradient for all negative inputs."]
    },
    {
        "id": "test_20",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "genomics",
        "claim_text": "The human mitochondrial chromosome spans over 500 million base pairs.",
        "quotes": ["Human mitochondrial DNA is a compact circular molecule comprising exactly 16,569 base pairs."]
    },
    {
        "id": "test_21",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "pharmacology",
        "claim_text": "Heparin directly breaks down pre-existing fibrin mesh clots.",
        "quotes": ["Heparin acts as an anticoagulant that accelerates antithrombin inhibition; it has no direct intrinsic fibrinolytic or clot-dissolving activity."]
    },
    {
        "id": "test_22",
        "label": "rejected",
        "phenomenon": "directional_negation",
        "domain": "biomedicine",
        "claim_text": "Thyroid hormone deficiency results in marked tachycardia and hyperthermia.",
        "quotes": ["Hypothyroidism leads to a hypometabolic state characteristically manifested by sinus bradycardia, cold intolerance, and hypothermia."]
    },
    {
        "id": "test_23",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "cell_biology",
        "claim_text": "Ribosomes synthesize polysaccharides through glycosidic bond formation.",
        "quotes": ["Ribosomes are macromolecular ribozyme machines dedicated exclusively to translating mRNA into polypeptide chains via peptide bond synthesis."]
    },
    {
        "id": "test_24",
        "label": "rejected",
        "phenomenon": "numerical_contradiction",
        "domain": "ai_ml",
        "claim_text": "The transformer training run was finished in 30 seconds on a single CPU core.",
        "quotes": ["Pre-training the baseline transformer required 14 days of continuous computation on an interconnected cluster of 256 A100 GPUs."]
    },
    {
        "id": "test_25",
        "label": "rejected",
        "phenomenon": "directional_negation",
        "domain": "genomics",
        "claim_text": "Nonsense mutations convert a stop codon into a sense amino acid codon.",
        "quotes": ["A nonsense mutation introduces a premature termination stop codon into the coding sequence, resulting in truncated polypeptide products."]
    },
    {
        "id": "test_26",
        "label": "rejected",
        "phenomenon": "universal_contradiction",
        "domain": "biomedicine",
        "claim_text": "Type 1 diabetes is primarily caused by excessive insulin hypersecretion.",
        "quotes": ["Type 1 diabetes results from autoimmune destruction of pancreatic beta cells, causing profound and absolute insulin deficiency."]
    },
    {
        "id": "test_27",
        "label": "rejected",
        "phenomenon": "direct_functional_negation",
        "domain": "pharmacology",
        "claim_text": "Aminoglycoside antibiotics kill bacteria by inhibiting cell wall peptidoglycan synthesis.",
        "quotes": ["Aminoglycosides exert bactericidal action by irreversibly binding the bacterial 30S ribosomal subunit to cause mistranslation and inhibit protein synthesis."]
    },

    # --- UNCERTAIN (13 cases) ---
    {
        "id": "test_28",
        "label": "uncertain",
        "phenomenon": "scope_overreach",
        "domain": "biomedicine",
        "claim_text": "The therapeutic monoclonal antibody prevents stroke in human hypertensive patients.",
        "quotes": ["In spontaneously hypertensive stroke-prone rat strains, prophylactic antibody treatment significantly reduced cerebral infarct size."]
    },
    {
        "id": "test_29",
        "label": "uncertain",
        "phenomenon": "causal_overreach",
        "domain": "cell_biology",
        "claim_text": "Telomere attrition directly causes neurodegenerative dementia.",
        "quotes": ["Shorter leukocyte telomere length was observed with higher frequency among elderly subjects diagnosed with progressive dementia."]
    },
    {
        "id": "test_30",
        "label": "uncertain",
        "phenomenon": "speculative_hedging",
        "domain": "genomics",
        "claim_text": "CRISPR off-target cleavage at non-canonical PAM sites causes oncogenic transformation.",
        "quotes": ["We hypothesize that non-canonical PAM cleavage events could conceivably trigger genomic instability under select selective pressures."]
    },
    {
        "id": "test_31",
        "label": "uncertain",
        "phenomenon": "quantitative_overreach",
        "domain": "pharmacology",
        "claim_text": "The new drug formulation increases bioavailability by exactly 3.5-fold.",
        "quotes": ["Pharmacokinetic curves demonstrated a statistically significant enhancement in systemic drug bioavailability over the standard pill."]
    },
    {
        "id": "test_32",
        "label": "uncertain",
        "phenomenon": "partial_support",
        "domain": "genomics",
        "claim_text": "The endonuclease cuts both strands cleanly and leaves four-base 3' single-stranded overhangs.",
        "quotes": ["Restriction digestion experiments confirmed that the enzyme cleaves both strands of double-stranded substrates cleanly."]
    },
    {
        "id": "test_33",
        "label": "uncertain",
        "phenomenon": "irrelevant_evidence",
        "domain": "genomics",
        "claim_text": "Cas9 recognizes target sites using an RNA-DNA heteroduplex loop.",
        "quotes": ["Superconducting quantum interference devices measure extremely subtle magnetic flux variations in cryogenic environments."]
    },
    {
        "id": "test_34",
        "label": "uncertain",
        "phenomenon": "outside_knowledge_trap",
        "domain": "genomics",
        "claim_text": "The structure of DNA is an antiparallel double helix.",
        "quotes": ["Bacteriophage T4 encodes its own DNA topoisomerase containing three essential gene products."]
    },
    {
        "id": "test_35",
        "label": "uncertain",
        "phenomenon": "missing_evidence",
        "domain": "pharmacology",
        "claim_text": "Monoclonal antibody conjugated with cytotoxic payload selectively targets tumor vasculature.",
        "quotes": []
    },
    {
        "id": "test_36",
        "label": "uncertain",
        "phenomenon": "scope_overreach",
        "domain": "pharmacology",
        "claim_text": "The antiviral candidate completely cures hepatitis C in human liver transplant recipients.",
        "quotes": ["In cell-free replicon assays, the novel polymerase inhibitor suppressed HCV RNA replication with an EC50 of 12 nM."]
    },
    {
        "id": "test_37",
        "label": "uncertain",
        "phenomenon": "causal_overreach",
        "domain": "biomedicine",
        "claim_text": "Chronic systemic inflammation directly triggers clinical major depressive disorder.",
        "quotes": ["Patients with major depressive disorder consistently exhibit elevated baseline circulating serum levels of inflammatory markers like IL-6 and CRP."]
    },
    {
        "id": "test_38",
        "label": "uncertain",
        "phenomenon": "speculative_hedging",
        "domain": "ai_ml",
        "claim_text": "Emergent abilities in large models arise from discrete phase transitions in optimization landscapes.",
        "quotes": ["It is possible that the sudden emergence of certain reasoning capabilities might be explained by non-linear metric effects rather than genuine phase transitions."]
    },
    {
        "id": "test_39",
        "label": "uncertain",
        "phenomenon": "quantitative_overreach",
        "domain": "ai_ml",
        "claim_text": "Flash attention decreases memory footprint by exactly 80 percent on all GPU architectures.",
        "quotes": ["By computing attention in tiles without materializing the full intermediate matrix, the algorithm significantly reduces GPU HBM memory requirements."]
    },
    {
        "id": "test_40",
        "label": "uncertain",
        "phenomenon": "missing_evidence",
        "domain": "cell_biology",
        "claim_text": "Nuclear pore complexes regulate bidirectional nucleocytoplasmic transport using disordered FG-repeat domains.",
        "quotes": []
    }
]


def build_datasets():
    dev_path = BENCHMARK_DIR / "dev.json"
    test_path = BENCHMARK_DIR / "test.json"

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(DEV_CASES, f, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(TEST_CASES, f, indent=2)

    print(f"[OK] Generated {len(DEV_CASES)} Dev cases at: {dev_path}")
    print(f"[OK] Generated {len(TEST_CASES)} Test cases at: {test_path}")

    # Verify balance
    for name, data in [("Dev", DEV_CASES), ("Test", TEST_CASES)]:
        counts = {}
        for item in data:
            counts[item["label"]] = counts.get(item["label"], 0) + 1
        print(f"  {name} Split Distribution: {counts} (Total: {len(data)})")


if __name__ == "__main__":
    build_datasets()
