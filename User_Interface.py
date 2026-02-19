import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg') # matplotlib im Hintergrundmodus für Streamlit
import matplotlib.pyplot as plt
from model import Structure

# speichert bei neu laden den alten Stand
if 'structure' not in st.session_state:
    st.session_state.structure = None
if 'last_result_fig' not in st.session_state:
    st.session_state.last_result_fig = None
if 'use_symmetry' not in st.session_state:
    st.session_state.use_symmetry = False
if 'constraints' not in st.session_state:
    st.session_state.constraints = []

st.set_page_config(page_title="FEM Optimierung", layout="wide")
st.title("FEM Optimierer")

# Sidebar----------------------------------------------------------
with st.sidebar:
    st.header("1. Geometrie")
    # Startwerte für Breite, Höhe, Auflösung und Material
    width = st.number_input("Gesamtbreite (m)", value=40.0, step=1.0)
    height = st.number_input("Gesamthöhe (m)", value=15.0, step=1.0)
    res = st.slider("Auflösung (m)", 0.5, 5.0, 1.5)
    st.divider()
    mat_type = st.selectbox("Material", ["Baustahl S235", "Aluminium", "Holz", "Custom"])
    e_mod_map = {"Baustahl S235": 210000.0, "Aluminium": 70000.0, "Holz": 10000.0, "Custom": 1000.0}
    e_modul = st.number_input("E-Modul (N/mm²)", value=e_mod_map[mat_type])
    
    if st.button("Gitter neu erzeugen", type="primary"):
        nx = int(width / res); nx = nx + 1 if nx % 2 == 0 else nx   # Kontenanzahl, immer ungerade für Symmetrie
        nz = int(height / res); nz = nz + 1 if nz % 2 == 0 else nz
        s = Structure(dim=2); s.generate_rect_mesh(nx, nz, width, height)   # Gitter erzeugen basierend auf Breite, Höhe und Auflösung
        for f in s.federn: f.k = e_modul / res
        st.session_state.structure = s; st.session_state.dims = (width, height); st.session_state.e_modul = e_modul # Model speichern
        st.session_state.last_result_fig = None; st.rerun()

# Plot Gitter Mash--------------------------------------------------------------------
def plot_with_stresses(structure, title, e_mod, w_orig, h_orig, vis_factor, is_setup_view=False, current_mass_pct=None, draw_sym_line=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-w_orig * 0.05, w_orig * 1.05)
    ax.set_ylim(-h_orig * 0.05, h_orig * 1.05)
    
    full_title = f"{title} (Material: {current_mass_pct:.1f}%)" if current_mass_pct is not None else title

    # Symmetrielinie
    if draw_sym_line:
        ax.axvline(x=w_orig/2, color='#0000FF', linestyle='--', alpha=0.8, lw=2, zorder=1)

    # Spannungen der Federn berechnen
    all_sigmas = []
    if not is_setup_view:
        for f in structure.federn:
            if f.massepunkt_i.active and f.massepunkt_j.active:
                L0 = np.linalg.norm(f.massepunkt_j.coords - f.massepunkt_i.coords)
                if L0 > 1e-9:
                    p1 = f.massepunkt_i.coords + f.massepunkt_i.displacement[:2]
                    p2 = f.massepunkt_j.coords + f.massepunkt_j.displacement[:2]
                    L1 = np.linalg.norm(p2 - p1)
                    all_sigmas.append(abs((L1 - L0) / L0) * e_mod)
    
    max_s = np.percentile(all_sigmas, 95) if all_sigmas and len(all_sigmas) > 5 else 1.0
    if max_s < 1e-3: max_s = 1.0

    cmap = plt.get_cmap('plasma')

    # Mash erzeugen
    for f in structure.federn:
        if is_setup_view or (f.massepunkt_i.active and f.massepunkt_j.active):
            p1 = f.massepunkt_i.coords + f.massepunkt_i.displacement[:2] * vis_factor
            p2 = f.massepunkt_j.coords + f.massepunkt_j.displacement[:2] * vis_factor
            
            # Layer 1: Skelett
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=0.5, alpha=0.2, zorder=1)

            color = '#A0A0A0'
            lw = 0.3
            alpha = 0.5

            if not is_setup_view:
                L0 = np.linalg.norm(f.massepunkt_j.coords - f.massepunkt_i.coords)
                sigma = 0
                if L0 > 1e-9:
                    p1_real = f.massepunkt_i.coords + f.massepunkt_i.displacement[:2]
                    p2_real = f.massepunkt_j.coords + f.massepunkt_j.displacement[:2]
                    sigma = abs((np.linalg.norm(p2_real-p1_real) - L0)/L0 * e_mod)
                
                color = cmap(min(sigma / max_s, 1.0))
                lw = 0.5 + (min(sigma / max_s, 1.0) * 3.0)
                alpha = 0.9

            # Layer 2: Farbe
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw, alpha=alpha, zorder=2)
    
    for m in structure.massepunkte:
        if is_setup_view or m.active:
            pos = m.coords + m.displacement[:2] * vis_factor
            if np.all(m.fixed): ax.plot(pos[0], pos[1], '^', color='red', ms=12, zorder=10)
            elif np.any(m.fixed): ax.plot(pos[0], pos[1], 'o', mfc='none', mec='orange', mew=2, ms=10, zorder=10)
            if np.linalg.norm(m.force) > 0:
                scale = (h * 0.15) / np.linalg.norm(m.force)
                dx = m.force[0] * scale
                dy = m.force[1] * scale
                ax.arrow(pos[0], pos[1], dx, dy, head_width=w*0.02, color='#00FF00', lw=2, zorder=20)
    
    if not is_setup_view:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('plasma'), norm=plt.Normalize(vmin=0, vmax=max_s))
        plt.colorbar(sm, ax=ax, label="Spannung [N/mm²]")

    ax.invert_yaxis(); ax.set_aspect('equal'); ax.set_title(full_title)
    return fig

def apply_constraints(struct):
    # 1. Alles zurücksetzen
    for m in struct.massepunkte: 
        m.fixed[:] = False
        m.force[:] = 0.0
    
    # 2. Liste aus Streamlit durchgehen
    for c in st.session_state.constraints:
        target = np.array([c['x'], c['z']]) # Sucht nächsten Knoten zum angegebenen Punkt
        node = min(struct.massepunkte, key=lambda m: np.linalg.norm(m.coords - target))
        
        if c['type'] == "Festlager":    # setzt Lager oder Kraft
            node.fixed[:] = True
        elif c['type'] == "Loslager": 
            node.fixed[1] = True 
        elif c['type'] == "Kraft":
            w_grad = c.get('angle', 270.0)
            w_rad = np.radians(w_grad)
            F = c['val']
            
            node.force[0] = F * np.cos(w_rad) 
            node.force[1] = -1.0 * F * np.sin(w_rad)

# User Interface--------------------------------------------------------------------
if st.session_state.structure:
    s, (w, h), e_mod = st.session_state.structure, st.session_state.dims, st.session_state.e_modul
    apply_constraints(s)

    st.subheader("Ausgangslage")
    
    # Plot der Ausgangslage mit Randbedingungen, Kräften und eventuell Symmetrielinie
    st.pyplot(plot_with_stresses(s, "Vorschau", e_mod, w, h, vis_factor=0, is_setup_view=True, draw_sym_line=st.session_state.use_symmetry))

    #tabs für Randbedingungen und Optimierung
    tab1, tab2 = st.tabs(["Randbedingungen", "Optimierung"])
    
    with tab1:
        c1, c2 = st.columns(2)
        def get_node(x, z):
            target = np.array([x, z])
            return min(s.massepunkte, key=lambda m: np.linalg.norm(m.coords - target))
        with c1:
            st.markdown("**Lager**")
            lx = st.number_input("Position X", 0.0, w, 0.0, key="lx_in", step=1.0)
            lz = st.number_input("Position Z", 0.0, h, h, key="lz_in", step=1.0)
            ltype = st.radio("Typ", ["Festlager (∆)", "Loslager (○)"], key="ltype_in")
            if st.button("Lager anwenden"):
                clean_type = "Festlager" if "Fest" in ltype else "Loslager"
                st.session_state.constraints.append({"type": clean_type, "x": lx, "z": lz})
                st.rerun()
        with c2:
            # Kraft mit Richtung und Betrag ändern // steps in 1.0 schritten
            st.markdown("**Kraft**")
            kx = st.number_input("Position X", 0.0, w, w/2, key="kx_in", step=1.0)
            kz = st.number_input("Position Z", 0.0, h, 0.0, key="kz_in", step=1.0)
            fv = st.number_input("Betrag (N)", value=5000.0, key="fv_in", step=500.0)
            fw = st.number_input("Winkel (°)", 0.0, 360.0, 270.0, step=45.0, key="fw_in")
            if st.button("Kraft anwenden"): 
                st.session_state.constraints.append({"type": "Kraft", "x": kx, "z": kz, "val": fv, "angle": fw}) 
                st.rerun()

        st.divider()
        # Liste der Randbedingungen
        st.write("**Aktuelle Lasten & Lager:**")

        if st.session_state.constraints:
            for i, c in enumerate(st.session_state.constraints):
                col_text, col_del = st.columns([4, 1])
                
                if c['type'] == "Kraft":
                    # Anzeige für KRAFT
                    info_text = f"**Kraft**: {c['val']} N ({c.get('angle', 270)}°) an Position ({c['x']}, {c['z']})"
                else:
                    # Anzeige für LAGER
                    symbol = "∆" if "Fest" in c['type'] else "○"
                    info_text = f"**{c['type']}** {symbol} an Position ({c['x']}, {c['z']})"

                col_text.markdown(info_text)
                
                # Löschen-Button
                if col_del.button("Löschen", key=f"delete_{i}"):
                    st.session_state.constraints.pop(i) # Eintrag entfernen
                    st.rerun() # Seite neu laden
        else:
            st.info("Die Liste ist leer. Füge oben Lager oder Kräfte hinzu.")
        st.divider()

        col1, col2 = st.columns([1,1])

        # Reset Button
        with col1:
            if st.button("Alles löschen (Reset)"):
                st.session_state.constraints = [] # Liste leeren!
                apply_constraints(s) # Modell leeren
                st.rerun()
        # Zu Optimierung wechseln
        with col2:
            st.text("Nun weiter zur Optimierung!")

    with tab2:
        # Checkbox Symetrie-Modus
        st.session_state.use_symmetry = st.checkbox("Symmetrie-Modus nutzen (Spiegelt Materialabtrag)", value=st.session_state.use_symmetry)

        target = st.slider("Ziel-Masse (%)", 5, 100, 50) / 100.0
        step = st.slider("Schrittweite", 0.01, 0.1, 0.02)
        vis = st.slider("Verformungs-Faktor", 1.0, 10.0, 1.0)
        
        if st.button("Optimierung starten", type="primary"):
            apply_constraints(s)
            
            # Testen ob Struktur stabil ist, bevor optimiert wird
            stable, message = s.stable_test()
            if not stable:
                st.error(f"Die Struktur ist instabil: {message} :(")
                st.stop()
            st.success("Die Struktur ist stabil! Starte Optimierung...")

            # Volle Struktur nutzen
            c_struct = s 
            for m in c_struct.massepunkte: m.active, m.displacement[:] = True, 0
            
            # Symmetrie-Partner vorberechnen
            partner_map = {} 
            if st.session_state.use_symmetry:
                nodes = c_struct.massepunkte
                for i, n in enumerate(nodes):
                    if n.coords[0] < w/2 - 0.01: # Links
                        target_x = w - n.coords[0]
                        target_z = n.coords[1]
                        dists = np.linalg.norm([p.coords - np.array([target_x, target_z]) for p in nodes], axis=1)
                        best_match_idx = np.argmin(dists)
                        if dists[best_match_idx] < 0.1: 
                            partner_map[i] = best_match_idx

            plot_spot = st.empty()
            curr_m = 1.0
            
            while curr_m > target:  # Rechne bis Zielmasse erreicht ist
                if c_struct.solve() is None: break  # Verschiebung berechnen
                c_struct.calculate_strain_energy()  # Energie berechnen
                
                # 1. Energie mitteln (links und rechts) für Symetrie
                if st.session_state.use_symmetry:
                    for idx_left, idx_right in partner_map.items():
                        avg_energy = (c_struct.massepunkte[idx_left].strain_energy + c_struct.massepunkte[idx_right].strain_energy) / 2
                        c_struct.massepunkte[idx_left].strain_energy = avg_energy
                        c_struct.massepunkte[idx_right].strain_energy = avg_energy
                
                # 2. Schutz für Massepunkte mit Kraft oder Randbedingung: Sehr hohe Energie zuweisen, damit sie nicht gelöscht werden
                for m in c_struct.massepunkte:
                    if np.linalg.norm(m.force) > 0 or np.any(m.fixed):
                        m.strain_energy = 1e15
                
                # 3. Löschen von Massepunkten mit geringster Energie je nach Schrittweite 1 bis 10% der Massepunkte
                c_struct.remove_inefficient_nodes(max(target, curr_m - step))

                # 4. Konstrolle der Symetrie
                if st.session_state.use_symmetry:
                    for idx_left, idx_right in partner_map.items():
                        # Wenn einer der beiden inaktiv ist, muss der andere es auch sein
                        if not c_struct.massepunkte[idx_left].active or not c_struct.massepunkte[idx_right].active:
                             c_struct.massepunkte[idx_left].active = False
                             c_struct.massepunkte[idx_right].active = False

                # 5. Plotten der aktuellen Struktur mit Spannungen
                curr_m = len([m for m in c_struct.massepunkte if m.active]) / len(c_struct.massepunkte)
                fig = plot_with_stresses(c_struct, "Optimierung", e_mod, w, h, vis, 
                                       draw_sym_line=st.session_state.use_symmetry, 
                                       current_mass_pct=curr_m*100)
                plot_spot.pyplot(fig); plt.close(fig)

            st.session_state.last_result_fig = fig