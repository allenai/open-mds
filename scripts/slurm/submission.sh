# MultiNews

# PRIMERA

# baseline
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/baseline.json"

# random
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/random/addition.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/random/deletion.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/random/replacement.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/random/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/random/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/random/backtranslation.json"

# best-case
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/best-case/addition.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/best-case/deletion.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/best-case/replacement.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/best-case/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/best-case/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/best-case/backtranslation.json"

# worst-case
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/addition.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/deletion.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/replacement.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/backtranslation.json"

# Pegasus

# baseline
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/baseline.json"

# random
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/random/addition.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/random/deletion.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/random/replacement.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/random/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/random/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/random/backtranslation.json"

# best-case
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/best-case/addition.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/best-case/deletion.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/best-case/replacement.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/best-case/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/best-case/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/best-case/backtranslation.json"

# worst-case
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/worst-case/addition.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/worst-case/deletion.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/worst-case/replacement.json"
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/worst-case/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/worst-case/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multi_news/pegasus/eval/worst-case/backtranslation.json"

# Multi-X-Science

# PRIMERA

# baseline
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/baseline.json"

# random
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/random/addition.json"
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/random/deletion.json"
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/random/replacement.json"
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/random/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/random/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/random/backtranslation.json"

# best-case
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/best-case/addition.json"
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/best-case/deletion.json"
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/best-case/replacement.json"
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/best-case/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/best-case/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multixscience/primera/eval/best-case/backtranslation.json"

# worst-case
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/addition.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/deletion.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/replacement.json"
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/duplication.json"
# Only need to call this once, not multiple times for different perturbed fractions!
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/sorting.json"
# Requires longer train time.
sbatch scripts/perturb.sh "conf/multi_news/primera/eval/worst-case/backtranslation.json"

