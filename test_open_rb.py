from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": False},
    experience="/home/seungjun/isaacsim/apps/isaacsim.exp.full.kit",
)

import omni.timeline
import omni.usd

RB_USD_PATH = "/home/seungjun/isaac_ws/Assets/rb5_850e_u.usd"

# 저장된 Stage 자체를 연다
result = omni.usd.get_context().open_stage(RB_USD_PATH)
print("open_stage result:", result)

# Stage 로딩 대기
for _ in range(30):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()

print("Stage root layer:", stage.GetRootLayer().identifier)
print("Default prim:", stage.GetDefaultPrim().GetPath())

print("\n========== Stage Prim Tree ==========")
for prim in stage.Traverse():
    print(prim.GetPath())
print("=====================================\n")

timeline = omni.timeline.get_timeline_interface()
timeline.play()

while simulation_app.is_running():
    simulation_app.update()

timeline.stop()
simulation_app.close()