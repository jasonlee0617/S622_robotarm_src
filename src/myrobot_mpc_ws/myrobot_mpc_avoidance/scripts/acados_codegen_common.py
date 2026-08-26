"""Shared local-toolbox setup and acados API compatibility patching."""

import os
import re
import sys
from pathlib import Path


def configure_local_toolbox():
    package_root = Path(__file__).resolve().parent.parent
    toolbox_dir = package_root.parent / "mpc_toolbox"
    acados_dir = toolbox_dir / "acados"
    casadi_dir = toolbox_dir / "python"
    template_dir = acados_dir / "interfaces" / "acados_template"

    if not template_dir.is_dir() or not (casadi_dir / "casadi").is_dir():
        raise RuntimeError(f"local acados/CasADi toolbox is incomplete under {toolbox_dir}.")

    os.chdir(package_root)
    os.environ["ACADOS_INSTALL_DIR"] = str(acados_dir)
    os.environ["ACADOS_SOURCE_DIR"] = str(acados_dir)
    os.environ["CASADI_DIR"] = str(casadi_dir)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    sys.path[:0] = [str(template_dir), str(casadi_dir)]
    return package_root, acados_dir, casadi_dir, template_dir


def verify_local_imports(acados_template, casadi, template_dir, casadi_dir):
    template_file = Path(acados_template.__file__).resolve()
    casadi_file = Path(casadi.__file__).resolve()
    if not template_file.is_relative_to(template_dir.resolve()):
        raise RuntimeError(f"Imported acados_template outside local toolbox: {template_file}")
    if not casadi_file.is_relative_to(casadi_dir.resolve()):
        raise RuntimeError(f"Imported CasADi outside local toolbox: {casadi_file}")
    return str(template_file), str(casadi_file)


def patch_generated_solver(package_root, acados_dir, generated_dir, model_name):
    """Patch only the generated wrapper when this acados C API requires it."""
    header = acados_dir / "include" / "acados_c" / "ocp_nlp_interface.h"
    if not header.exists():
        return

    header_text = header.read_text(encoding="utf-8", errors="ignore")
    needs_new_signature = (
        "ocp_nlp_in *in, ocp_nlp_out *out, int stage" in header_text
        and "ocp_nlp_out *out, ocp_nlp_in *in" in header_text
        and "ocp_nlp_out *out, const char *field" in header_text
    )
    if not needs_new_signature:
        return

    solver_c = package_root / generated_dir / f"acados_solver_{model_name}.c"
    text = solver_c.read_text(encoding="utf-8")
    original = text
    if not re.search(r"ocp_nlp_out\s*\*\s*nlp_out\s*=\s*capsule->nlp_out;", text):
        text = text.replace(
            "    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;\n\n    int tmp_int = 0;",
            "    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;\n"
            "    ocp_nlp_out* nlp_out = capsule->nlp_out;\n\n"
            "    int tmp_int = 0;",
        )
    text = re.sub(
        r"ocp_nlp_constraints_model_set\(nlp_config, nlp_dims, nlp_in, (?!nlp_out, )",
        "ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, ", text)
    text = re.sub(
        r"ocp_nlp_out_set\(nlp_config, nlp_dims, nlp_out, (?!capsule->nlp_in, |nlp_in, )",
        "ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, capsule->nlp_in, ", text)
    text = text.replace(
        "    // 4) create nlp_in\n"
        "    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);\n"
        "\n"
        "    // 5) setup functions, nlp_in and default parameters",
        "    // 4) create nlp_in and nlp_out\n"
        "    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);\n"
        "    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);\n"
        "    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);\n"
        "\n"
        "    // 5) setup functions, nlp_in and default parameters",
    )
    text = text.replace(
        "    // 7) create and set nlp_out\n"
        "    // 7.1) nlp_out\n"
        "    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);\n"
        "    // 7.2) sens_out\n"
        "    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);\n"
        f"    {model_name}_acados_set_nlp_out(capsule);",
        "    // 7) initialize nlp_out\n"
        f"    {model_name}_acados_set_nlp_out(capsule);",
    )
    text = re.sub(
        r"ocp_nlp_dims_get_total_from_attr\((capsules\[0\]->nlp_solver->config,\s*"
        r"capsules\[0\]->nlp_solver->dims),\s*field\)",
        r"ocp_nlp_dims_get_total_from_attr(\1, capsules[0]->nlp_out, field)", text)
    if text != original:
        solver_c.write_text(text, encoding="utf-8")
        print("Patched generated acados solver for current ocp_nlp_interface API.")
