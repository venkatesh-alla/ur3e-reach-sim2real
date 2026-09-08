import csv
from pathlib import Path

class EvaluationLogger:

    def __init__(self, model_name: str, results_dir: str = "real/eval_results"):

        self.base_dir = (
            Path(results_dir)
            / model_name
        )

        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.goals_file = self.base_dir / "goals.csv"
        self.episode_results_file = self.base_dir / "episode_results.csv"
        self.seed_summary_file = self.base_dir / "seed_summary.csv"
        self.final_results_file = self.base_dir / "final_results.csv"

        self._initialize_files()

    # --------------------------------------------------------
    # Create CSV files with headers if they do not exist
    # --------------------------------------------------------

    def _initialize_files(self):

        if not self.goals_file.exists():

            with open(self.goals_file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "evaluation_seed",
                    "episode",
                    "goal_x",
                    "goal_y",
                    "goal_z",
                ])

        if not self.episode_results_file.exists():

            with open(self.episode_results_file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "evaluation_seed",
                    "episode",
                    "reward",
                    "length",
                    "success",
                ])

        if not self.seed_summary_file.exists():

            with open(self.seed_summary_file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "evaluation_seed",
                    "num_episodes",
                    "mean_reward",
                    "mean_length",
                    "success_rate",
                ])

        if not self.final_results_file.exists():

            with open(self.final_results_file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "overall_mean_reward",
                    "overall_mean_length",
                    "overall_success_rate",
                ])

    # --------------------------------------------------------
    # Goal logging
    # --------------------------------------------------------

    def log_goal(
        self,
        evaluation_seed,
        episode,
        goal,
    ):

        with open(self.goals_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                evaluation_seed,
                episode,
                float(goal[0]),
                float(goal[1]),
                float(goal[2]),
            ])

    # --------------------------------------------------------
    # Episode-level logging
    # --------------------------------------------------------

    def log_episode_result(
        self,
        evaluation_seed,
        episode,
        reward,
        length,
        success,
    ):

        with open(self.episode_results_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                evaluation_seed,
                episode,
                reward,
                length,
                success,
            ])

    # --------------------------------------------------------
    # Seed summary logging
    # --------------------------------------------------------

    def log_seed_summary(
        self,
        evaluation_seed,
        num_episodes,
        mean_reward,
        mean_length,
        success_rate,
    ):

        with open(self.seed_summary_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                evaluation_seed,
                num_episodes,
                mean_reward,
                mean_length,
                success_rate,
            ])

    # --------------------------------------------------------
    # Final aggregate results
    # --------------------------------------------------------

    def log_final_results(
        self,
        overall_mean_reward,
        overall_mean_length,
        overall_success_rate,
    ):

        with open(self.final_results_file, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "overall_mean_reward",
                "overall_mean_length",
                "overall_success_rate",
            ])

            writer.writerow([
                overall_mean_reward,
                overall_mean_length,
                overall_success_rate,
            ])
