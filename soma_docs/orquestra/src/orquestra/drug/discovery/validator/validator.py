import numpy as np
import tqdm

from .chem_filters import GeneralFilter, PainFilter, WehiMCFilter
from .filter_abstract import FilterAbstract

_DEFAULT_CONDITION_FILTERS = [GeneralFilter(), PainFilter(), WehiMCFilter()]
_DEFAULT_WEIGHT_FILTERS = [1.0, 1.0, 1.0]


class ConditionFilters:
    def __init__(
        self,
        filter_lists: list[FilterAbstract] | None = None,
        weight_lists: list[float] | None = None,
    ) -> None:
        """
        Args:
            filter_lists: list of filters. If None, use default filters.
        """
        self.filter_lists = filter_lists or _DEFAULT_CONDITION_FILTERS
        self.all_results = []
        self.weight_lists = weight_lists or _DEFAULT_WEIGHT_FILTERS

    def apply_all(self, smiles):
        u_smiles = list(set(smiles))

        all_returns = np.zeros(len(u_smiles))
        all_passed_smile = []
        self.all_rewards = np.zeros(len(u_smiles))
        i = 0
        with tqdm.tqdm(total=len(u_smiles)) as pbar:
            for smile in u_smiles:
                pbar.set_description(
                    f"Filtered {i} / {len(u_smiles)}. passed={int(all_returns.sum())},"
                    f"frac={int(all_returns.sum())/len(u_smiles)}"
                )
                all_results = []
                for filter in self.filter_lists:
                    try:
                        all_results.append(filter.apply(smile))
                    except Exception:
                        all_results.append(False)

                if all(all_results):
                    all_passed_smile.append(smile)

                all_returns[i] = all(all_results)
                self.all_rewards[i] = np.sum(
                    np.array(all_results).astype(int) * self.weight_lists
                )
                i += 1
                pbar.update()
            self.all_returns = all_returns
            self.all_passed_smile = all_passed_smile
        return all_returns

    def compute_reward(self, smile):
        all_results = []
        for filter in self.filter_lists:
            try:
                all_results.append(filter.apply(smile))
            except Exception:
                all_results.append(False)

        filter_value = all(all_results)
        reward_value = np.sum(np.array(all_results).astype(int) * self.weight_lists)

        return (
            filter_value,
            reward_value,
            all_results,
            np.array(all_results).astype(int) * self.weight_lists,
        )

    def get_validity_smiles(self, smiles):
        return 100 * np.sum(self.apply_all(smiles)) / len(smiles)
