# Templates for benchmarking

def directional_relation_mc(object):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Where is the {object} relative to your current position
and heading?

[OPTIONS]
(A) To your left
(B) To your right
(C) In front of you
(D) Behind you

[GROUND TRUTH] Computed from bearing angle between agent heading
and object centroid:
  front:  -45° to +45°
  right:   45° to 135°
  behind: 135° to 225° / -135° to -180°
  left:   225° to 315° / -45° to -135°"""

def directional_relation_binary(object, direction):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Is the {object} to your {direction}?

[OPTIONS]
(A) Yes
(B) No

[GROUND TRUTH] True if object falls within the angular bin
for {direction}, False otherwise.
NOTE: balance positive and negative examples 50/50
across your dataset to avoid answer bias."""

def directional_relation_two_objects(object1, object2, direction):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Which object is more to your {direction} —
the {object1} or the {object2}?

[OPTIONS]
(A) The {object1}
(B) The {object2}
(C) They are at approximately the same position

[GROUND TRUTH] Compare the signed angular component
(lateral for left/right, depth for front/behind)
of each object relative to agent heading."""

def vertical_relation(object1, object2):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Is the {object1} above or below the {object2}
from your current viewpoint?

[OPTIONS]
(A) Above
(B) Below
(C) At approximately the same height

[GROUND TRUTH] Compare Y-axis (vertical) coordinates
of object centroids in THOR world space.
Threshold for "same height": difference < 0.1m."""

def angular_relation(object):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Imagine a clock face centered on you,
with 12 o'clock directly ahead of you.
At which clock position is the {object}?

[OPTIONS]
(A) Around 12 o'clock (directly ahead)
(B) Around 3 o'clock (to your right)
(C) Around 6 o'clock (directly behind)
(D) Around 9 o'clock (to your left)

[GROUND TRUTH] Bearing angle from agent heading
to object centroid, mapped to clock positions:
  12 o'clock: -45° to +45°
   3 o'clock:  45° to 135°
   6 o'clock: 135° to 225°
   9 o'clock: 225° to 315°
"""

def eight_way_angular_position(object):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Imagine a clock face centered on you,
with 12 o'clock directly ahead.
Which best describes the position of the {object}?

[OPTIONS]
(A) 12 o'clock — directly ahead
(B) 1–2 o'clock — front-right
(C) 3 o'clock — directly right
(D) 4–5 o'clock — back-right
(E) 6 o'clock — directly behind
(F) 7–8 o'clock — back-left
(G) 9 o'clock — directly left
(H) 10–11 o'clock — front-left

[GROUND TRUTH] Bearing angle binned into
eight 45° sectors. Use this template only
for objects that are clearly within one sector,
not near sector boundaries (angular distance
from nearest boundary > 10°)."""

def eight_way_angular_position_2objs(object1, object2):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Which object is more directly in front of you —
the {object1} or the {object2}?

[OPTIONS]
(A) The {object1}
(B) The {object2}
(C) Both are at approximately the same angle ahead

[GROUND TRUTH] Compare absolute angular deviation
from agent heading for each object.
The one with smaller absolute bearing angle
is more directly in front.
Threshold for "same angle": difference < 5°."""

def distance_relation(object):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Is the {object} close to you or at a distance?

[OPTIONS]
(A) Close (within arm's reach)
(B) At a distance (further away but still visible)

[GROUND TRUTH] Euclidean distance from agent to object centroid.
  Close:      < 0.75m
  At distance: 0.75m to 1.5m
Filter: exclude objects within 0.1m of the 0.75m boundary."""

def distante_relation_2objs(object1, object2):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Which is closer to you — the {object1}
or the {object2}?

[OPTIONS]
(A) The {object1}
(B) The {object2}
(C) They are approximately the same distance

[GROUND TRUTH] Compare Euclidean distances.
Threshold for "same distance": difference < 0.15m.
Filter: only generate if distance difference > 0.15m
to avoid ambiguous cases."""

def distance_relation_mc(object1, object2, object3):
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] Order these objects from nearest to farthest:
{object1}, {object2}, {object3}.

[OPTIONS]
(A) {object1}, {object2}, {object3}
(B) {object1}, {object3}, {object2}
(C) {object2}, {object1}, {object3}
(D) {object2}, {object3}, {object1}
(E) {object3}, {object1}, {object2}
(F) {object3}, {object2}, {object1}

[GROUND TRUTH] Sort by Euclidean distance.
Filter: only generate if all pairwise distance
differences > 0.15m to ensure clear ordering."""

def count_objects():
    return f"""[CONTEXT] You are an agent navigating an indoor environment.
The image shows your current egocentric view.

[QUESTION] How many objects are within very close range
(less than 0.75 meters) of your current position?

[OPTIONS]
(A) None
(B) One
(C) Two
(D) Three or more

[GROUND TRUTH] Count objects with distance < 0.75m
from THOR metadata.
"""