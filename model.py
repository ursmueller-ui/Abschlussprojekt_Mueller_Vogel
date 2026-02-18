import numpy as np
from dataclasses import dataclass, field
from solver import solve as linear_solve 

@dataclass
class Massepunkt:
    id: int             # eindeutige ID
    coords: np.ndarray  # Pos. [x, y, (z)]
    fixed: np.ndarray   # Abhängigkeiten (z.B. Festlager) [True, False]
    force: np.ndarray   # Kraftvektor [Fx, Fy, (Fz)]
    active: bool = True # False -> gelöscht
    mass: float = 1.0   # Standardmasse 1kg
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(3))   # Verschiebungsvektor
    strain_energy: float = 0.0  #Verformungsenergie

class Feder:
    def __init__(self, massepunkt_i: Massepunkt, massepunkt_j: Massepunkt, k_val: float = 1.0):
        self.massepunkt_i = massepunkt_i    # Verbundene Massepunkte i und j
        self.massepunkt_j = massepunkt_j
        self.k = k_val                      # Federsteifigkeit (N/m)

    def get_element_stiffness(self, dim: int) -> np.ndarray:
        """
        Berechnet Steifigkeits- und Transformationsmatrix für 2D und 3D!
        """
        # Richtungsvektor berechnen 
        vec = self.massepunkt_j.coords - self.massepunkt_i.coords
        length = np.linalg.norm(vec)
        
        if length == 0:
            return np.zeros((2*dim, 2*dim)) # Vermeidung von Division durch Null -> gibt Matrix mit Nullen zurück
            
        e_n = vec / length # Einheitsvektor
        
        # Transformationsmatrix O = e_n * e_n
        O = np.outer(e_n, e_n)
        
        # Basis-Steifigkeit
        K_base = self.k * np.array([[1, -1], [-1, 1]])
        
        # Kronecker-Produkt für die globale Orientierung
        # Das erzeugt autom. die richtige Matrixgröße (4x4 für 2D, 6x6 für 3D)
        K_local = np.kron(K_base, O)
        
        return K_local  # Steifigkeits- mit Transformationsmatrix

class Structure:
    def __init__(self, dim: int = 2):
        self.massepunkte = []    # Alle Massepunkte der Struktur
        self.federn = []         # Alle Federn der Struktur
        self.dim = dim           # 2D oder 3D

    def add_Massepunkt(self, x, z, y=0.0):
        # Erstellt Massepunkt passend zur Dimension
        if self.dim == 2:
            # Im 2D-Fall speichern wir [x, z]
            coords = np.array([x, z])
        else:
            # Im 3D-Fall speichern wir [x, y, z]
            coords = np.array([x, y, z])
        # Initial keine Kraft, nicht fixiert
        fixed = np.zeros(self.dim, dtype=bool)
        force = np.zeros(self.dim)
        
        new_massepunkt = Massepunkt(len(self.massepunkte), coords, fixed, force)
        self.massepunkte.append(new_massepunkt)
        return new_massepunkt

    def add_Feder(self, idx_i, idx_j, k=1.0):
        # neue Feder zwischen Massepunkt i und j erstellen
        el = Feder(self.massepunkte[idx_i], self.massepunkte[idx_j], k)
        self.federn.append(el)
        
    def assemble_system(self):
            # Steifigkeitsmatrix K und Kraftvektor F aufbauen
            n_dof = len(self.massepunkte) * self.dim
            K_g = np.zeros((n_dof, n_dof))
            F_g = np.zeros(n_dof)
            
            # 1. Globale Steifigkeitsmatrix aufbauen
            for feder in self.federn:
                # Nur aktive Federn berücksichtigen!
                if not feder.massepunkt_i.active or not feder.massepunkt_j.active:
                    continue

                # Lokale Matrix holen (automatisch 2D oder 3D)
                k_local = feder.get_element_stiffness(self.dim)
                
                # IDs der beteiligten Knoten
                i = feder.massepunkt_i.id
                j = feder.massepunkt_j.id
                
                # Liste aller betroffenen Knotenindizes in der globalen Matrix:
                # Indizes für Knoten i (Start) und j (Ende)
                idx_i = list(range(i * self.dim, i * self.dim + self.dim))
                idx_j = list(range(j * self.dim, j * self.dim + self.dim))
                
                # Alle Indizes zusammen (das entspricht den Zeilen/Spalten in k_local)
                idxs = idx_i + idx_j
                
                # Wir addieren die lokale Matrix auf die Stellen in der globalen Matrix.
                K_g[np.ix_(idxs, idxs)] += k_local

            # 2. Kraftvektor aufbauen
            # Wir gehen alle Punkte durch und schauen, ob eine Kraft wirkt
            for node in self.massepunkte:
                # Wo startet der Eintrag für diesen Knoten im Vektor?
                start_idx = node.id * self.dim
                
                # Kraft eintragen (Fx, Fy, (Fz))
                F_g[start_idx : start_idx + self.dim] = node.force

            return K_g, F_g
    
    def solve(self):
        """Löst das System K*u=F und speichert Ergebnisse in den Knoten."""
      
        # 1. Matrix und Vektor bauen
        K, F = self.assemble_system()
        
        # 2. Fixierte Freiheitsgrade finden (Randbedingungen)
        fixed_dofs = []
        for m in self.massepunkte:
            for d in range(self.dim):
                if m.fixed[d]:
                    # Globaler Index = ID * Dimension + Dimension_Index (0=x, 1=y, 2=z)
                    fixed_dofs.append(m.id * self.dim + d)
        
        # 3. Solver aufrufen
        u_vec = linear_solve(K, F, fixed_dofs)
        
        if u_vec is None:   # Fehlerbehandlung für singuläre Systeme (nicht eindeutig lösbar)
            print("System ist singulär/nicht lösbar!")
            return None
            
        # 4. Ergebnis zurück in die Massepunkte schreiben
        for m in self.massepunkte:
            start = m.id * self.dim
            end = start + self.dim
            m.displacement = u_vec[start:end]
            
        return u_vec
    
    def calculate_strain_energy(self):

        """
        Berechnet die Verformungsenergie für alle aktiven Federn
        und verteilt sie 50/50 auf die Massenpunkte.
        """

        # 1. Energie aller Knoten zurücksetzen
        for m in self.massepunkte:
            m.strain_energy = 0.0

        # 2. Über alle Federn iterieren
        for feder in self.federn:
            # Ignoriere inaktive Federn
            if not feder.massepunkt_i.active or not feder.massepunkt_j.active:
                continue

            # Vektoren und Matrizen holen
            k_local = feder.get_element_stiffness(self.dim)
            
            # Verschiebeungsvektor u für dieses Element bauen
            u_i = feder.massepunkt_i.displacement[:self.dim]
            u_j = feder.massepunkt_j.displacement[:self.dim]
            u_element = np.concatenate([u_i, u_j])

            # Matrix-Multiplikation: 0.5 * u^T * K * u
            # (u @ K) -> Vektor-Matrix-Multiplikation
            # dot(u) -> Skalarprodukt am Ende
            energy = 0.5 * np.dot(u_element @ k_local, u_element)

            # Energie 50/50 aufteilen
            feder.massepunkt_i.strain_energy += energy * 0.5
            feder.massepunkt_j.strain_energy += energy * 0.5

    def remove_inefficient_nodes(self, target_mass_percent: float):
        """
        Deaktiviert die Knoten mit dem geringsten Kraftfluss, bis die Zielmasse erreicht ist.
        target_mass_percent: z.B. 0.5 für 50% der Masse übrig lassen.
        """
        # Nur momentan aktive Knoten betrachten
        active_nodes = [m for m in self.massepunkte if m.active]
        current_count = len(active_nodes)
        target_count = int(len(self.massepunkte) * target_mass_percent)
        
        # Wie viele Massenpunkte müssen weg?
        to_remove_count = current_count - target_count
        
        if to_remove_count <= 0:
            return # Ziel schon erreicht
            
        candidates = []
        for m in active_nodes:
            # Knoten mit Kraft oder Randbedingung dürfen nicht gelöscht werden!
            is_fixed = np.any(m.fixed)
            has_force = np.linalg.norm(m.force) > 1e-9
            
            if not is_fixed and not has_force:
                candidates.append(m)
        
        # Aufsteigend sortieren nach Energie
        candidates.sort(key=lambda m: m.strain_energy)
        
        # Konnten mit geringster Energie zuerst entfernen
        num_delete = min(to_remove_count, len(candidates))
        
        for i in range(num_delete):
            candidates[i].active = False
            
        print(f"Optimierung: {num_delete} Knoten entfernt.")
    
    def generate_rect_mesh(self, nx: int, nz: int, width: float, height: float):
        """
        Erstellt ein Rechteck-Gitter mit nx * nz Knoten.
        """
        # 1. Alte Daten löschen
        self.massepunkte = []
        self.federn = []
        
        # Abstände berechnen
        dx = width / (nx - 1) if nx > 1 else 0
        dz = height / (nz - 1) if nz > 1 else 0
        
        # 2. Knoten erstellen
        for z_i in range(nz):
            for x_i in range(nx):
                # Koordinaten: x geht nach rechts, z nach unten
                self.add_Massepunkt(x_i * dx, z = z_i * dz)

        # 3. Federn erstellen
        # Wir gehen durch jeden Punkt und verbinden ihn mit seinen Nachbarn
        # (Rechts, Unten, Unten-Rechts, Unten-Links)
        
        k_ortho = 1.0
        k_diag = 1.0 / np.sqrt(2.0)
        
        for z_i in range(nz):
            for x_i in range(nx):
                # Aktueller Index im 1D-Array
                idx = z_i * nx + x_i
                
                # --- Nachbar Rechts (Horizontal) ---
                if x_i < nx - 1:
                    idx_right = idx + 1
                    self.add_Feder(idx, idx_right, k=k_ortho)
                
                # --- Nachbar Unten (Vertikal) ---
                if z_i < nz - 1:
                    idx_down = (z_i + 1) * nx + x_i
                    self.add_Feder(idx, idx_down, k=k_ortho)
                    
                # --- Nachbar Unten-Rechts (Diagonal) ---
                if x_i < nx - 1 and z_i < nz - 1:
                    idx_br = (z_i + 1) * nx + (x_i + 1)
                    self.add_Feder(idx, idx_br, k=k_diag)
                    
                # --- Nachbar Unten-Links (Diagonal) ---
                if x_i > 0 and z_i < nz - 1:
                    idx_bl = (z_i + 1) * nx + (x_i - 1)
                    self.add_Feder(idx, idx_bl, k=k_diag)
                    
    def stable_test(self):

        #Freiheitsgrade und Lager überprüfen, damit die Optimierung durchgeführt werden kann.
        fixed_nodes = 0
        fixed_dofs = 0
        festlager = 0
        loslager = 0

        for m in self.massepunkte:
            if np.any(m.fixed):
                fixed_nodes += 1
                n_fixed = np.sum(m.fixed)
                fixed_dofs += n_fixed

                if n_fixed == self.dim:
                    festlager += 1
                else:
                    loslager += 1

        # Verschiedene Fehlermeldungen ausgeben, je nach Fehler
        required_dofs = 3 if self.dim == 2 else 6

        if fixed_nodes < 2:
            return False, "Mindestens 2 Lager werden gebraucht!"

        if fixed_dofs < required_dofs:
            return False, f"Es werden mindestens {required_dofs} fixierte Freiheitsgrade benötigt!"

        if festlager < 1:
            return False, "Mindestens 1 Festlager wird benötigt!"

        if loslager < 1:
            return False, "Mindestens 1 Loslager wird benötigt!"

        return True, "Struktur stabil."