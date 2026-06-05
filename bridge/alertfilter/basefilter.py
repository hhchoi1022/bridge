from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from bridge.configuration import Configuration
from bridge.utils import HostGalaxyCatalog


class BaseFilter(ABC):
    def __init__(self, config_file: str, hostgalaxycatalog: HostGalaxyCatalog = None):
        self.config = Configuration(config_filenames=[config_file])

        if hostgalaxycatalog is None:
            self.hostgalaxycatalog = HostGalaxyCatalog()
        else:
            self.hostgalaxycatalog = hostgalaxycatalog

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def apply(self, tbl: Table, return_all=False, verbose=True):

        tbl = self.apply_catalog_filter(tbl, return_all=return_all, verbose=verbose)
        if len(tbl) == 0:
            return tbl

        tbl = self.apply_hostgalaxy_filter(tbl, return_all=return_all, verbose=verbose)
        # if len(tbl) == 0:
        #     return tbl

        # tbl = self.apply_magnitude_filter(tbl, return_all=return_all, verbose=verbose)
        return tbl

    # -----------------------------
    # ABSTRACT METHODS
    # -----------------------------
    @abstractmethod
    def apply_hostgalaxy_filter(self, tbl: Table, return_all: bool = False, verbose=True):
        pass

    @abstractmethod
    def apply_magnitude_filter(self, tbl: Table, return_all: bool = False, verbose=True):
        pass

        # -----------------------------
        # COMMON: catalog filter
        # -----------------------------
    # def apply_catalog_filter(self, tbl: Table, return_all: bool = False, verbose=True):

    #     filtered_tbl = tbl.copy()
    #     len_all = len(filtered_tbl)

    #     total_mask = np.ones(len_all, dtype=bool)

    #     for key, constraint in tqdm(
    #         self.config.catalog_constraints.items(),
    #         desc='Applying catalog filter...'
    #     ):

    #         if key not in filtered_tbl.colnames:
    #             print(f"WARNING: {key} not in catalog columns. Skipping...")
    #             continue

    #         col_mask = np.ones(len_all, dtype=bool)

    #         if isinstance(constraint, (list, tuple)):
    #             for single_constraint in constraint:
    #                 mask = self._apply_single_constraint(filtered_tbl, key, single_constraint)
    #                 col_mask &= mask
    #         else:
    #             col_mask &= self._apply_single_constraint(filtered_tbl, key, constraint)

    #         filtered_tbl[f"mask_{key}"] = col_mask

    #         # ✅ 핵심: cumulative mask
    #         prev_count = np.sum(total_mask)
    #         total_mask &= col_mask
    #         new_count = np.sum(total_mask)

    #         if verbose:
    #             print(f'Filter {key}: {prev_count} -> {new_count}')

    #     filtered_tbl = self._update_mask_all(filtered_tbl)

    #     return filtered_tbl if return_all else filtered_tbl[total_mask]

    def apply_catalog_filter(self, tbl: Table, return_all: bool = False, verbose=True):

        filtered_tbl = tbl.copy()
        len_all = len(filtered_tbl)

        total_mask = np.ones(len_all, dtype=bool)

        for key, constraint in tqdm(
            self.config.catalog_constraints.items(),
            desc='Applying catalog filter...'
        ):

            if key not in filtered_tbl.colnames:
                print(f"WARNING: {key} not in catalog columns. Skipping...")
                continue

            if isinstance(constraint, dict):
                if 'and' in constraint:
                    col_mask = np.ones(len_all, dtype=bool)
                    for single_constraint in constraint['and']:
                        mask = self._apply_single_constraint(filtered_tbl, key, single_constraint)
                        col_mask &= mask

                elif 'or' in constraint:
                    col_mask = np.zeros(len_all, dtype=bool)
                    for single_constraint in constraint['or']:
                        mask = self._apply_single_constraint(filtered_tbl, key, single_constraint)
                        col_mask |= mask

                else:
                    print(f"WARNING: Unsupported constraint format for {key}: {constraint}")
                    continue

            elif isinstance(constraint, (list, tuple)):
                # backward compatibility: list means AND
                col_mask = np.ones(len_all, dtype=bool)
                for single_constraint in constraint:
                    mask = self._apply_single_constraint(filtered_tbl, key, single_constraint)
                    col_mask &= mask

            else:
                col_mask = self._apply_single_constraint(filtered_tbl, key, constraint)

            filtered_tbl[f"mask_{key}"] = col_mask

            prev_count = np.sum(total_mask)
            total_mask &= col_mask
            new_count = np.sum(total_mask)

            if verbose:
                print(f'Filter {key}: {prev_count} -> {new_count}')

        filtered_tbl = self._update_mask_all(filtered_tbl)

        return filtered_tbl if return_all else filtered_tbl[total_mask]
    
    def _update_mask_all(self, tbl):
        mask_cols = [col for col in tbl.colnames if col.startswith('mask_')]

        if len(mask_cols) == 0:
            return tbl

        mask_all = np.ones(len(tbl), dtype=bool)
        for col in mask_cols:
            mask_all &= np.asarray(tbl[col], dtype=bool)

        tbl['mask_all'] = mask_all
        return tbl

    # -----------------------------
    # COMMON: constraint parser
    # -----------------------------
    def _apply_single_constraint(self, tbl: Table, key: str, raw_value):

        if raw_value is None:
            return np.ones(len(tbl), dtype=bool)

        value = str(raw_value).strip()
        if value == "":
            return np.ones(len(tbl), dtype=bool)

        # ------------------------------
        # parse operator
        # ------------------------------
        op = None
        rhs_str = None

        for candidate in ("<=", ">=", "!=", "==", "<", ">", "="):
            if value.startswith(candidate):
                op = candidate
                rhs_str = value[len(candidate):].strip()
                break

        if op is None:
            op = "=="
            rhs_str = value

        col = tbl[key]
        col_arr = np.asarray(col)

        # =========================================================
        # 🔥 1. BOOLEAN HANDLING (추가된 핵심)
        # =========================================================
        if np.issubdtype(col_arr.dtype, np.bool_):

            if op in ("<", "<=", ">", ">="):
                print(
                    f"WARNING: Column '{key}' is boolean but uses comparison '{op}'. Skipping..."
                )
                return np.ones(len(tbl), dtype=bool)

            if isinstance(raw_value, (bool, np.bool_)):
                rhs = bool(raw_value)
            else:
                s = rhs_str.lower()
                if s in ("true", "1", "yes", "y"):
                    rhs = True
                elif s in ("false", "0", "no", "n"):
                    rhs = False
                else:
                    print(
                        f"WARNING: Cannot parse boolean '{rhs_str}' for column '{key}'. Skipping..."
                    )
                    return np.ones(len(tbl), dtype=bool)

            if op in ("=", "=="):
                return col_arr == rhs
            elif op == "!=":
                return col_arr != rhs
            else:
                return np.ones(len(tbl), dtype=bool)

        # =========================================================
        # 2. NUMERIC / STRING HANDLING (기존 로직 유지)
        # =========================================================
        try:
            rhs = float(rhs_str)
            col_arr = col_arr.astype(float)
        except:
            rhs = rhs_str

        if op == "<":
            return col_arr < rhs
        elif op == "<=":
            return col_arr <= rhs
        elif op == ">":
            return col_arr > rhs
        elif op == ">=":
            return col_arr >= rhs
        elif op in ("=", "=="):
            return col_arr == rhs
        elif op == "!=":
            return col_arr != rhs

        return np.ones(len(tbl), dtype=bool)