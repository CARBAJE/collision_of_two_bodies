from __future__ import annotations

from typing import Any, Iterable, Optional
import time 

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Visualizer:
    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

    def animate_3d(
        self,
        trajectories: list[np.ndarray], 
        interval_ms: int = 50,
        title: str = "Simulación de Órbita en 3D",
        total_frames: int = 300,
    ) -> None:
        """
        Crea y muestra una animación de las trayectorias en 3D.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib/NumPy no disponible, se omite la visualización.")
            return
        
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d') 

        
        ax.set_title(title)
        ax.set_xlabel("Eje X")
        ax.set_ylabel("Eje Y")
        ax.set_zlabel("Eje Z")
        ax.grid(True)
        
        
        all_data = np.concatenate(trajectories)
        max_range = np.array([
            all_data[:, 0].max() - all_data[:, 0].min(),
            all_data[:, 1].max() - all_data[:, 1].min(),
            all_data[:, 2].max() - all_data[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_data[:, 0].max() + all_data[:, 0].min()) * 0.5
        mid_y = (all_data[:, 1].max() + all_data[:, 1].min()) * 0.5
        mid_z = (all_data[:, 2].max() + all_data[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        
        
        for traj_idx, traj in enumerate(trajectories):
            ax.plot(
                traj[:total_frames, 0], 
                traj[:total_frames, 1], 
                traj[:total_frames, 2], 
                linestyle='--', 
                alpha=0.3, 
                label=f'Órbita {traj_idx+1}'
            )
        
        
        
        points = [ax.plot([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], marker='o', markersize=5, label=f'Cuerpo {i+1}')[0]
                  for i, traj in enumerate(trajectories)]

        
        def update_frame(frame):
            for i, (point, traj) in enumerate(zip(points, trajectories)):
                
                x, y, z = traj[frame, 0], traj[frame, 1], traj[frame, 2]
                point.set_data_3d(x, y, z)
                
                
                
                
            ax.set_title(f"{title}\nTiempo: {frame}/{total_frames}")
            return points 
        
        
        num_frames = min(total_frames, min(len(t) for t in trajectories))
        ani = animation.FuncAnimation(
            fig, 
            update_frame, 
            frames=num_frames, 
            interval=interval_ms, 
            blit=False, 
            repeat=True 
        )

        
        if not self.headless:
            
            plt.show()
        else:
            plt.close()