"""
Comprehensive Research Validation Framework for Publication-Ready BCI Studies.

This module implements a complete research validation framework with experimental
design, statistical analysis, reproducibility protocols, and publication-ready
result generation for academic BCI research contributions.

Research Contributions:
- Rigorous experimental design with power analysis and effect size calculations
- Comprehensive statistical validation with multiple comparison corrections
- Reproducibility protocols with version control and data provenance
- Automated benchmarking against state-of-the-art methods
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Protocol
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import time
import hashlib
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare, kruskal
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalDesign:
    """Configuration for experimental design and validation."""
    
    # Study design
    study_type: str = "comparative"  # "comparative", "ablation", "longitudinal"
    n_subjects: int = 20
    n_sessions_per_subject: int = 5
    n_trials_per_session: int = 100
    
    # Statistical parameters
    alpha: float = 0.05  # Significance level
    power: float = 0.8   # Statistical power
    effect_size: float = 0.5  # Expected effect size (Cohen's d)
    multiple_comparison_method: str = "fdr_bh"  # "bonferroni", "fdr_bh", "holm"
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "subject_wise"  # "random", "subject_wise", "time_series"
    n_repetitions: int = 10
    
    # Permutation testing
    use_permutation_tests: bool = True
    n_permutations: int = 1000
    
    # Reproducibility
    random_seed: int = 42
    save_intermediate_results: bool = True
    track_data_provenance: bool = True
    
    # Benchmarking
    baseline_methods: List[str] = field(default_factory=lambda: ["svm", "lda", "cnn"])
    benchmark_datasets: List[str] = field(default_factory=lambda: ["bci_competition", "physionet"])
    
    # Publication requirements
    generate_figures: bool = True
    create_supplementary_material: bool = True
    export_latex_tables: bool = True


@dataclass 
class ValidationResult:
    """Container for validation results."""
    
    experiment_id: str
    timestamp: str
    study_design: ExperimentalDesign
    
    # Performance metrics
    accuracy: Dict[str, float] = field(default_factory=dict)
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)
    auc_roc: Dict[str, float] = field(default_factory=dict)
    
    # Statistical tests
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Cross-validation results
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    cv_mean: Dict[str, float] = field(default_factory=dict)
    cv_std: Dict[str, float] = field(default_factory=dict)
    
    # Reproducibility info
    data_hash: str = ""
    code_version: str = ""
    system_info: Dict[str, str] = field(default_factory=dict)
    
    # Publication materials
    figures: Dict[str, str] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    supplementary_data: Dict[str, Any] = field(default_factory=dict)


class BenchmarkMethod(Protocol):
    """Protocol for benchmark methods."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the method to training data."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on test data."""
        ...
    
    def get_name(self) -> str:
        """Get method name."""
        ...


class ResearchValidator:
    """Main research validation framework."""
    
    def __init__(self, design: ExperimentalDesign, output_dir: str = "./validation_results"):
        self.design = design
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(design.random_seed)
        torch.manual_seed(design.random_seed)
        
        # Initialize benchmark methods
        self.benchmark_methods = self._initialize_benchmark_methods()
        
        # Results storage
        self.validation_results = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_benchmark_methods(self) -> Dict[str, BenchmarkMethod]:
        """Initialize benchmark methods for comparison."""
        methods = {}
        
        if "svm" in self.design.baseline_methods:
            from sklearn.svm import SVC
            methods["SVM"] = SVC(kernel='rbf', random_state=self.design.random_seed)
        
        if "lda" in self.design.baseline_methods:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            methods["LDA"] = LinearDiscriminantAnalysis()
        
        if "cnn" in self.design.baseline_methods:
            # Placeholder for CNN implementation
            methods["CNN"] = DummyClassifier("CNN")
        
        if "random_forest" in self.design.baseline_methods:
            from sklearn.ensemble import RandomForestClassifier
            methods["RandomForest"] = RandomForestClassifier(
                n_estimators=100, random_state=self.design.random_seed
            )
        
        return methods
    
    def validate_method(
        self,
        method: BenchmarkMethod,
        data: Dict[str, Any],
        experiment_name: str
    ) -> ValidationResult:
        """
        Comprehensive validation of a BCI method.
        
        Args:
            method: Method to validate
            data: Dataset with X_train, y_train, X_test, y_test, subject_ids
            experiment_name: Name for this experiment
            
        Returns:
            Complete validation results
        """
        experiment_id = f"{experiment_name}_{int(time.time())}"
        self.logger.info(f"Starting validation for {experiment_name}")
        
        # Initialize result container
        result = ValidationResult(
            experiment_id=experiment_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            study_design=self.design,
            data_hash=self._compute_data_hash(data),
            code_version=self._get_code_version(),
            system_info=self._get_system_info()
        )
        
        # Step 1: Power analysis
        power_analysis = self._perform_power_analysis(data)
        result.supplementary_data["power_analysis"] = power_analysis
        
        # Step 2: Cross-validation evaluation
        cv_results = self._cross_validation_evaluation(method, data)
        result.cv_scores = cv_results["scores"]
        result.cv_mean = cv_results["means"]
        result.cv_std = cv_results["stds"]
        
        # Step 3: Performance metrics
        performance_metrics = self._compute_performance_metrics(method, data)
        result.accuracy = performance_metrics["accuracy"]
        result.precision = performance_metrics["precision"]
        result.recall = performance_metrics["recall"]
        result.f1_score = performance_metrics["f1_score"]
        result.auc_roc = performance_metrics["auc_roc"]
        
        # Step 4: Statistical significance testing
        statistical_results = self._statistical_significance_testing(method, data)
        result.statistical_tests = statistical_results["tests"]
        result.effect_sizes = statistical_results["effect_sizes"]
        result.confidence_intervals = statistical_results["confidence_intervals"]
        
        # Step 5: Benchmark comparison
        benchmark_results = self._benchmark_comparison(method, data)
        result.supplementary_data["benchmark_comparison"] = benchmark_results
        
        # Step 6: Generate publication materials
        if self.design.generate_figures:
            figures = self._generate_figures(result, data)
            result.figures = figures
        
        if self.design.export_latex_tables:
            tables = self._generate_latex_tables(result)
            result.tables = tables
        
        if self.design.create_supplementary_material:
            supplementary = self._create_supplementary_material(result, data)
            result.supplementary_data.update(supplementary)
        
        # Save results
        self._save_results(result)
        
        self.logger.info(f"Validation completed for {experiment_name}")
        return result
    
    def _perform_power_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical power analysis."""
        self.logger.info("Performing power analysis")
        
        # Sample sizes
        n_samples = len(data["y_train"])
        n_subjects = len(np.unique(data.get("subject_ids", np.arange(n_samples))))
        
        # Power analysis for t-test
        required_n = ttest_power(
            effect_size=self.design.effect_size,
            power=self.design.power,
            alpha=self.design.alpha,
            alternative='two-sided'
        )
        
        # Achieved power with current sample size
        achieved_power = ttest_power(
            effect_size=self.design.effect_size,
            nobs=n_subjects,
            alpha=self.design.alpha,
            alternative='two-sided'
        )
        
        power_analysis = {
            "current_sample_size": n_samples,
            "n_subjects": n_subjects,
            "required_sample_size": required_n,
            "achieved_power": achieved_power,
            "target_power": self.design.power,
            "effect_size": self.design.effect_size,
            "alpha": self.design.alpha,
            "adequately_powered": achieved_power >= self.design.power
        }
        
        self.logger.info(f"Power analysis: achieved power = {achieved_power:.3f}")
        return power_analysis
    
    def _cross_validation_evaluation(self, method: BenchmarkMethod, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive cross-validation."""
        self.logger.info("Performing cross-validation evaluation")
        
        X = data["X_train"]
        y = data["y_train"]
        subject_ids = data.get("subject_ids", None)
        
        cv_results = {"scores": {}, "means": {}, "stds": {}}
        
        # Different CV strategies
        cv_strategies = []
        
        if self.design.cv_strategy == "random":
            from sklearn.model_selection import KFold
            cv_strategies.append(("Random CV", KFold(n_splits=self.design.cv_folds, shuffle=True, 
                                                   random_state=self.design.random_seed)))
        
        elif self.design.cv_strategy == "subject_wise" and subject_ids is not None:
            from sklearn.model_selection import GroupKFold
            cv_strategies.append(("Subject-wise CV", GroupKFold(n_splits=min(self.design.cv_folds, 
                                                               len(np.unique(subject_ids))))))
        
        elif self.design.cv_strategy == "time_series":
            from sklearn.model_selection import TimeSeriesSplit
            cv_strategies.append(("Time Series CV", TimeSeriesSplit(n_splits=self.design.cv_folds)))
        
        # Default to random CV if no valid strategy
        if not cv_strategies:
            from sklearn.model_selection import KFold
            cv_strategies.append(("Random CV", KFold(n_splits=self.design.cv_folds, shuffle=True, 
                                                   random_state=self.design.random_seed)))
        
        # Perform CV for each strategy
        for cv_name, cv_splitter in cv_strategies:
            scores_list = []
            
            # Multiple repetitions
            for rep in range(self.design.n_repetitions):
                # Set different seed for each repetition
                np.random.seed(self.design.random_seed + rep)
                
                if hasattr(method, 'random_state'):
                    method.random_state = self.design.random_seed + rep
                
                # Perform cross-validation
                if subject_ids is not None and cv_name == "Subject-wise CV":
                    scores = cross_val_score(method, X, y, cv=cv_splitter, groups=subject_ids, 
                                           scoring='accuracy', n_jobs=-1)
                else:
                    scores = cross_val_score(method, X, y, cv=cv_splitter, scoring='accuracy', n_jobs=-1)
                
                scores_list.extend(scores)
            
            cv_results["scores"][cv_name] = scores_list
            cv_results["means"][cv_name] = np.mean(scores_list)
            cv_results["stds"][cv_name] = np.std(scores_list)
            
            self.logger.info(f"{cv_name}: {np.mean(scores_list):.3f} ± {np.std(scores_list):.3f}")
        
        return cv_results
    
    def _compute_performance_metrics(self, method: BenchmarkMethod, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Compute comprehensive performance metrics."""
        self.logger.info("Computing performance metrics")
        
        # Train on full training set
        method.fit(data["X_train"], data["y_train"])
        
        # Predict on test set
        y_pred = method.predict(data["X_test"])
        y_true = data["y_test"]
        
        # Get prediction probabilities if available
        y_prob = None
        if hasattr(method, 'predict_proba'):
            y_prob = method.predict_proba(data["X_test"])
        elif hasattr(method, 'decision_function'):
            y_prob = method.decision_function(data["X_test"])
            if y_prob.ndim == 1:
                y_prob = np.column_stack([-y_prob, y_prob])
        
        # Compute metrics
        metrics = {
            "accuracy": {"overall": accuracy_score(y_true, y_pred)},
            "precision": {},
            "recall": {},
            "f1_score": {},
            "auc_roc": {}
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        unique_classes = np.unique(y_true)
        for i, class_label in enumerate(unique_classes):
            metrics["precision"][f"class_{class_label}"] = precision_per_class[i]
            metrics["recall"][f"class_{class_label}"] = recall_per_class[i]
            metrics["f1_score"][f"class_{class_label}"] = f1_per_class[i]
        
        # Macro and weighted averages
        metrics["precision"]["macro"] = precision_score(y_true, y_pred, average='macro')
        metrics["precision"]["weighted"] = precision_score(y_true, y_pred, average='weighted')
        metrics["recall"]["macro"] = recall_score(y_true, y_pred, average='macro')
        metrics["recall"]["weighted"] = recall_score(y_true, y_pred, average='weighted')
        metrics["f1_score"]["macro"] = f1_score(y_true, y_pred, average='macro')
        metrics["f1_score"]["weighted"] = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC
        if y_prob is not None:
            if len(unique_classes) == 2:
                metrics["auc_roc"]["overall"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics["auc_roc"]["macro"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics["auc_roc"]["weighted"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        
        return metrics
    
    def _statistical_significance_testing(self, method: BenchmarkMethod, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        self.logger.info("Performing statistical significance testing")
        
        results = {
            "tests": {},
            "effect_sizes": {},
            "confidence_intervals": {}
        }
        
        # Permutation test for classification accuracy
        if self.design.use_permutation_tests:
            X = data["X_train"]
            y = data["y_train"]
            
            # Compute permutation test score
            score, perm_scores, p_value = permutation_test_score(
                method, X, y, scoring='accuracy', cv=self.design.cv_folds,
                n_permutations=self.design.n_permutations, n_jobs=-1,
                random_state=self.design.random_seed
            )
            
            results["tests"]["permutation_test"] = {
                "score": score,
                "p_value": p_value,
                "significant": p_value < self.design.alpha,
                "permutation_scores": perm_scores.tolist()
            }
            
            # Effect size (standardized difference)
            effect_size = (score - np.mean(perm_scores)) / np.std(perm_scores)
            results["effect_sizes"]["permutation_effect"] = effect_size
            
            # Confidence interval
            ci_lower = np.percentile(perm_scores, (self.design.alpha/2) * 100)
            ci_upper = np.percentile(perm_scores, (1 - self.design.alpha/2) * 100)
            results["confidence_intervals"]["permutation_ci"] = (ci_lower, ci_upper)
        
        # Cross-validation statistical test
        cv_scores = self.validation_results.get("cv_scores", {})
        if cv_scores:
            for cv_type, scores in cv_scores.items():
                if len(scores) > 1:
                    # One-sample t-test against chance level
                    chance_level = 1.0 / len(np.unique(data["y_train"]))
                    t_stat, p_value = stats.ttest_1samp(scores, chance_level)
                    
                    results["tests"][f"{cv_type}_ttest"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < self.design.alpha,
                        "chance_level": chance_level
                    }
                    
                    # Cohen's d effect size
                    cohens_d = (np.mean(scores) - chance_level) / np.std(scores)
                    results["effect_sizes"][f"{cv_type}_cohens_d"] = cohens_d
                    
                    # Confidence interval for mean
                    sem = stats.sem(scores)
                    ci = stats.t.interval(1 - self.design.alpha, len(scores) - 1, 
                                        loc=np.mean(scores), scale=sem)
                    results["confidence_intervals"][f"{cv_type}_mean_ci"] = ci
        
        return results
    
    def _benchmark_comparison(self, method: BenchmarkMethod, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare against benchmark methods."""
        self.logger.info("Performing benchmark comparison")
        
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        
        # Evaluate all methods
        method_scores = {}
        method_scores[method.get_name()] = self._evaluate_single_method(method, X_train, y_train, X_test, y_test)
        
        for name, benchmark_method in self.benchmark_methods.items():
            method_scores[name] = self._evaluate_single_method(benchmark_method, X_train, y_train, X_test, y_test)
        
        # Statistical comparison
        comparisons = {}
        main_method_name = method.get_name()
        
        for benchmark_name in self.benchmark_methods.keys():
            # McNemar's test for paired classification results
            main_correct = method_scores[main_method_name]["predictions"] == y_test
            benchmark_correct = method_scores[benchmark_name]["predictions"] == y_test
            
            # Create contingency table
            both_correct = np.sum(main_correct & benchmark_correct)
            main_only = np.sum(main_correct & ~benchmark_correct)
            benchmark_only = np.sum(~main_correct & benchmark_correct)
            both_wrong = np.sum(~main_correct & ~benchmark_correct)
            
            contingency_table = np.array([[both_correct, main_only], [benchmark_only, both_wrong]])
            
            # McNemar's test
            mcnemar_result = mcnemar(contingency_table, exact=False)
            
            comparisons[f"vs_{benchmark_name}"] = {
                "main_method_accuracy": method_scores[main_method_name]["accuracy"],
                "benchmark_accuracy": method_scores[benchmark_name]["accuracy"],
                "accuracy_difference": method_scores[main_method_name]["accuracy"] - method_scores[benchmark_name]["accuracy"],
                "mcnemar_statistic": mcnemar_result.statistic,
                "mcnemar_p_value": mcnemar_result.pvalue,
                "significant_difference": mcnemar_result.pvalue < self.design.alpha,
                "contingency_table": contingency_table.tolist()
            }
        
        # Multiple comparison correction
        if len(comparisons) > 1:
            p_values = [comp["mcnemar_p_value"] for comp in comparisons.values()]
            corrected_p_values = multipletests(p_values, alpha=self.design.alpha, 
                                             method=self.design.multiple_comparison_method)[1]
            
            for i, (comp_name, comp_data) in enumerate(comparisons.items()):
                comp_data["corrected_p_value"] = corrected_p_values[i]
                comp_data["significant_after_correction"] = corrected_p_values[i] < self.design.alpha
        
        return {
            "method_scores": method_scores,
            "statistical_comparisons": comparisons
        }
    
    def _evaluate_single_method(self, method: BenchmarkMethod, X_train: np.ndarray, 
                               y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate a single method."""
        method.fit(X_train, y_train)
        predictions = method.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            "accuracy": accuracy,
            "predictions": predictions
        }
    
    def _generate_figures(self, result: ValidationResult, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate publication-ready figures."""
        self.logger.info("Generating publication figures")
        
        figures = {}
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # Figure 1: Cross-validation results
        fig, ax = plt.subplots(figsize=(10, 6))
        cv_means = list(result.cv_mean.values())
        cv_stds = list(result.cv_std.values())
        cv_names = list(result.cv_mean.keys())
        
        x_pos = np.arange(len(cv_names))
        bars = ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        ax.set_xlabel('Cross-Validation Strategy')
        ax.set_ylabel('Accuracy')
        ax.set_title('Cross-Validation Performance')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cv_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, cv_means, cv_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        cv_fig_path = self.output_dir / f"{result.experiment_id}_cv_results.png"
        plt.savefig(cv_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        figures["cross_validation"] = str(cv_fig_path)
        
        # Figure 2: Performance comparison
        if "benchmark_comparison" in result.supplementary_data:
            benchmark_data = result.supplementary_data["benchmark_comparison"]
            method_scores = benchmark_data["method_scores"]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            methods = list(method_scores.keys())
            accuracies = [scores["accuracy"] for scores in method_scores.values()]
            
            bars = ax.bar(methods, accuracies, alpha=0.8)
            ax.set_xlabel('Methods')
            ax.set_ylabel('Accuracy')
            ax.set_title('Benchmark Comparison')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            benchmark_fig_path = self.output_dir / f"{result.experiment_id}_benchmark.png"
            plt.savefig(benchmark_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            figures["benchmark_comparison"] = str(benchmark_fig_path)
        
        # Figure 3: Statistical significance visualization
        if result.statistical_tests:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            # Permutation test histogram
            if "permutation_test" in result.statistical_tests:
                perm_data = result.statistical_tests["permutation_test"]
                perm_scores = perm_data["permutation_scores"]
                actual_score = perm_data["score"]
                
                axes[0].hist(perm_scores, bins=50, alpha=0.7, density=True, label='Permutation scores')
                axes[0].axvline(actual_score, color='red', linestyle='--', label=f'Actual score: {actual_score:.3f}')
                axes[0].set_xlabel('Accuracy')
                axes[0].set_ylabel('Density')
                axes[0].set_title(f'Permutation Test (p={perm_data["p_value"]:.4f})')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Effect sizes
            if result.effect_sizes:
                effect_names = list(result.effect_sizes.keys())
                effect_values = list(result.effect_sizes.values())
                
                bars = axes[1].bar(range(len(effect_names)), effect_values, alpha=0.8)
                axes[1].set_xlabel('Effect Type')
                axes[1].set_ylabel('Effect Size')
                axes[1].set_title('Effect Sizes')
                axes[1].set_xticks(range(len(effect_names)))
                axes[1].set_xticklabels(effect_names, rotation=45, ha='right')
                axes[1].grid(True, alpha=0.3)
                
                # Add Cohen's d interpretation lines
                axes[1].axhline(0.2, color='green', linestyle=':', alpha=0.5, label='Small effect')
                axes[1].axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Medium effect')
                axes[1].axhline(0.8, color='red', linestyle=':', alpha=0.5, label='Large effect')
                axes[1].legend(fontsize=8)
            
            # Confidence intervals
            if result.confidence_intervals:
                ci_names = list(result.confidence_intervals.keys())
                ci_values = list(result.confidence_intervals.values())
                
                means = [np.mean(ci) for ci in ci_values]
                errors = [(ci[1] - ci[0]) / 2 for ci in ci_values]
                
                axes[2].errorbar(range(len(ci_names)), means, yerr=errors, 
                               fmt='o', capsize=5, capthick=2)
                axes[2].set_xlabel('Metric')
                axes[2].set_ylabel('Value')
                axes[2].set_title('Confidence Intervals')
                axes[2].set_xticks(range(len(ci_names)))
                axes[2].set_xticklabels(ci_names, rotation=45, ha='right')
                axes[2].grid(True, alpha=0.3)
            
            # Remove unused subplot
            axes[3].remove()
            
            plt.tight_layout()
            stats_fig_path = self.output_dir / f"{result.experiment_id}_statistics.png"
            plt.savefig(stats_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            figures["statistical_analysis"] = str(stats_fig_path)
        
        return figures
    
    def _generate_latex_tables(self, result: ValidationResult) -> Dict[str, pd.DataFrame]:
        """Generate LaTeX-formatted tables for publication."""
        self.logger.info("Generating LaTeX tables")
        
        tables = {}
        
        # Performance metrics table
        perf_data = []
        for metric_type in ["accuracy", "precision", "recall", "f1_score"]:
            metric_dict = getattr(result, metric_type, {})
            for metric_name, value in metric_dict.items():
                perf_data.append({
                    "Metric": f"{metric_type.title()} ({metric_name})",
                    "Value": f"{value:.3f}"
                })
        
        perf_df = pd.DataFrame(perf_data)
        tables["performance_metrics"] = perf_df
        
        # Cross-validation results table
        cv_data = []
        for cv_type in result.cv_mean.keys():
            cv_data.append({
                "CV Strategy": cv_type,
                "Mean Accuracy": f"{result.cv_mean[cv_type]:.3f}",
                "Std Deviation": f"{result.cv_std[cv_type]:.3f}",
                "95% CI": f"[{result.cv_mean[cv_type] - 1.96*result.cv_std[cv_type]:.3f}, "
                         f"{result.cv_mean[cv_type] + 1.96*result.cv_std[cv_type]:.3f}]"
            })
        
        cv_df = pd.DataFrame(cv_data)
        tables["cross_validation"] = cv_df
        
        # Statistical tests table
        stats_data = []
        for test_name, test_result in result.statistical_tests.items():
            stats_data.append({
                "Test": test_name.replace('_', ' ').title(),
                "Statistic": f"{test_result.get('score', test_result.get('t_statistic', 'N/A')):.4f}",
                "p-value": f"{test_result.get('p_value', 'N/A'):.4f}",
                "Significant": "Yes" if test_result.get('significant', False) else "No"
            })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            tables["statistical_tests"] = stats_df
        
        # Save tables as LaTeX
        for table_name, df in tables.items():
            latex_path = self.output_dir / f"{result.experiment_id}_{table_name}.tex"
            latex_str = df.to_latex(index=False, escape=False, float_format='%.3f')
            
            with open(latex_path, 'w') as f:
                f.write(latex_str)
            
            # Also save as CSV for easy viewing
            csv_path = self.output_dir / f"{result.experiment_id}_{table_name}.csv"
            df.to_csv(csv_path, index=False)
        
        return tables
    
    def _create_supplementary_material(self, result: ValidationResult, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create supplementary material for publication."""
        self.logger.info("Creating supplementary material")
        
        supplementary = {}
        
        # Data characteristics
        supplementary["data_characteristics"] = {
            "n_samples_train": len(data["X_train"]),
            "n_samples_test": len(data["X_test"]),
            "n_features": data["X_train"].shape[1],
            "n_classes": len(np.unique(data["y_train"])),
            "class_distribution_train": {
                str(label): int(count) for label, count in 
                zip(*np.unique(data["y_train"], return_counts=True))
            },
            "class_distribution_test": {
                str(label): int(count) for label, count in 
                zip(*np.unique(data["y_test"], return_counts=True))
            }
        }
        
        # Detailed CV results
        supplementary["detailed_cv_results"] = result.cv_scores
        
        # Effect size interpretations
        effect_interpretations = {}
        for effect_name, effect_value in result.effect_sizes.items():
            if abs(effect_value) < 0.2:
                interpretation = "negligible"
            elif abs(effect_value) < 0.5:
                interpretation = "small"
            elif abs(effect_value) < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            effect_interpretations[effect_name] = {
                "value": effect_value,
                "interpretation": interpretation
            }
        
        supplementary["effect_size_interpretations"] = effect_interpretations
        
        # Reproducibility information
        supplementary["reproducibility"] = {
            "random_seed": self.design.random_seed,
            "data_hash": result.data_hash,
            "system_info": result.system_info,
            "study_design": {
                "n_cv_folds": self.design.cv_folds,
                "n_repetitions": self.design.n_repetitions,
                "alpha_level": self.design.alpha,
                "multiple_comparison_correction": self.design.multiple_comparison_method
            }
        }
        
        return supplementary
    
    def _compute_data_hash(self, data: Dict[str, Any]) -> str:
        """Compute hash of dataset for reproducibility tracking."""
        # Concatenate all data arrays
        data_str = ""
        for key in sorted(data.keys()):
            if isinstance(data[key], np.ndarray):
                data_str += str(data[key].tobytes())
            else:
                data_str += str(data[key])
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_code_version(self) -> str:
        """Get code version for reproducibility."""
        # In practice, this would get git commit hash or version tag
        return "v1.0.0"  # Placeholder
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for reproducibility."""
        import platform
        import sys
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__
        }
    
    def _save_results(self, result: ValidationResult):
        """Save validation results to file."""
        # Save as JSON
        result_dict = result.__dict__.copy()
        
        # Convert non-serializable objects
        for key, value in result_dict.items():
            if isinstance(value, np.ndarray):
                result_dict[key] = value.tolist()
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        result_dict[key][subkey] = subvalue.tolist()
        
        json_path = self.output_dir / f"{result.experiment_id}_results.json"
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save as pickle for full object preservation
        pickle_path = self.output_dir / f"{result.experiment_id}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        
        self.logger.info(f"Results saved to {json_path} and {pickle_path}")


class DummyClassifier(BenchmarkMethod):
    """Dummy classifier for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Simple majority class classifier
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.majority_class)
    
    def get_name(self) -> str:
        return self.name


class MetaAnalyzer:
    """Meta-analysis across multiple studies."""
    
    def __init__(self):
        self.studies = []
        self.logger = logging.getLogger(__name__)
    
    def add_study(self, result: ValidationResult):
        """Add a study to meta-analysis."""
        self.studies.append(result)
    
    def perform_meta_analysis(self) -> Dict[str, Any]:
        """Perform meta-analysis across studies."""
        if len(self.studies) < 2:
            raise ValueError("Meta-analysis requires at least 2 studies")
        
        self.logger.info(f"Performing meta-analysis on {len(self.studies)} studies")
        
        # Collect effect sizes and sample sizes
        effect_sizes = []
        sample_sizes = []
        study_names = []
        
        for study in self.studies:
            # Use first available effect size
            if study.effect_sizes:
                effect_size = list(study.effect_sizes.values())[0]
                effect_sizes.append(effect_size)
                
                # Estimate sample size from CV results
                if study.cv_scores:
                    cv_scores = list(study.cv_scores.values())[0]
                    sample_sizes.append(len(cv_scores))
                else:
                    sample_sizes.append(100)  # Default
                
                study_names.append(study.experiment_id)
        
        if not effect_sizes:
            raise ValueError("No effect sizes found in studies")
        
        effect_sizes = np.array(effect_sizes)
        sample_sizes = np.array(sample_sizes)
        
        # Fixed-effects meta-analysis
        weights = sample_sizes
        weighted_mean = np.average(effect_sizes, weights=weights)
        
        # Random-effects meta-analysis (simplified)
        Q = np.sum(weights * (effect_sizes - weighted_mean)**2)
        df = len(effect_sizes) - 1
        tau_squared = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
        
        re_weights = 1 / (1/weights + tau_squared)
        re_weighted_mean = np.average(effect_sizes, weights=re_weights)
        
        # Heterogeneity statistics
        I_squared = max(0, (Q - df) / Q * 100) if Q > 0 else 0
        
        return {
            "n_studies": len(self.studies),
            "study_names": study_names,
            "individual_effect_sizes": effect_sizes.tolist(),
            "individual_sample_sizes": sample_sizes.tolist(),
            "fixed_effect_size": weighted_mean,
            "random_effect_size": re_weighted_mean,
            "heterogeneity_q": Q,
            "heterogeneity_i_squared": I_squared,
            "tau_squared": tau_squared
        }


def create_research_validation_framework(
    design: Optional[ExperimentalDesign] = None,
    output_dir: str = "./validation_results"
) -> ResearchValidator:
    """
    Create a research validation framework.
    
    Args:
        design: Experimental design configuration
        output_dir: Output directory for results
        
    Returns:
        Configured research validator
    """
    if design is None:
        design = ExperimentalDesign(
            study_type="comparative",
            n_subjects=20,
            alpha=0.05,
            power=0.8,
            use_permutation_tests=True,
            generate_figures=True,
            export_latex_tables=True
        )
    
    validator = ResearchValidator(design, output_dir)
    
    logger.info(f"Created research validation framework with {design.study_type} design")
    
    return validator


# Example usage for BCI research validation
def validate_bci_method_example():
    """Example of how to validate a BCI method for publication."""
    
    # Create synthetic BCI dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 64
    n_classes = 2
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Add some signal to make it non-random
    X[y == 1, :10] += 0.5  # Class 1 has higher signal in first 10 features
    
    # Split data
    split_idx = int(0.8 * n_samples)
    data = {
        "X_train": X[:split_idx],
        "y_train": y[:split_idx],
        "X_test": X[split_idx:],
        "y_test": y[split_idx:],
        "subject_ids": np.repeat(range(10), n_samples // 10)
    }
    
    # Create experimental design
    design = ExperimentalDesign(
        study_type="comparative",
        n_subjects=10,
        cv_folds=5,
        use_permutation_tests=True,
        generate_figures=True,
        export_latex_tables=True
    )
    
    # Create validator
    validator = create_research_validation_framework(design)
    
    # Create dummy method for testing
    class DummyBCIMethod(BenchmarkMethod):
        def __init__(self):
            from sklearn.svm import SVC
            self.model = SVC(kernel='rbf', random_state=42)
        
        def fit(self, X, y):
            self.model.fit(X, y)
        
        def predict(self, X):
            return self.model.predict(X)
        
        def get_name(self):
            return "Novel BCI Method"
    
    # Validate method
    method = DummyBCIMethod()
    results = validator.validate_method(method, data, "novel_bci_experiment")
    
    return results