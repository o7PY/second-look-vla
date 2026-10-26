"""Safe destruction of robosuite render contexts when several LIBERO envs live in one process (found 2026-10-10 while testing G1; tests/p2/test_live_loop.py).

robosuite's `MjRenderContext.__del__` frees the MuJoCo render context with `mjr_freeContext` in WHATEVER EGL context is current, then destroys its own EGL context and
calls `eglReleaseThread()`, which also un-currents the thread. When a context is destroyed late (garbage collection after a hard reset or after `Scene.close()`), while
another env's context is current, that deletes the other env's GL objects and leaves the thread without a current context: later renders come out as striped garbage
(policy observations of the second episode, with a render Scene alive). Here `__del__` first makes its OWN context current, frees both, and then restores the context
that was current before. Idempotent; apply before any env is built (`second_look.loop.Rig` does).
"""
from __future__ import annotations


def patch() -> None:
    from mujoco.egl import egl_ext as EGL
    from robosuite.renderers.context import egl_context as E
    from robosuite.utils import binding_utils as bu
    cls = bu.MjRenderContext
    if getattr(cls, "_sl_safe_del", False):
        return

    def safe_del(self):
        prev = own = None
        try:
            prev = EGL.eglGetCurrentContext()
            own = getattr(self.gl_ctx, "_context", None)
            self.gl_ctx.make_current()
            self.con.free()
            self.gl_ctx.free()
        except Exception:
            pass
        finally:
            try:
                if prev and (own is None or prev.address != own.address):
                    EGL.eglMakeCurrent(E.EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, prev)
            except Exception:
                pass

    cls.__del__ = safe_del
    cls._sl_safe_del = True
