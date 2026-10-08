"""The plain-torch reference backend: the correctness oracle, CPU-capable.

Reference-only ops (the spherical harmonics) live here. They are closed-form
and cheap, are never dispatched to an accelerated kernel today, and are written
for clarity rather than speed.
"""
