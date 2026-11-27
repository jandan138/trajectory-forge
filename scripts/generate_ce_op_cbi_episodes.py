#!/usr/bin/env python3
"""Generate deterministic C+B+I episode metadata variants for CE-OP Bench synthesis.

Given a `scenario-forge` exported `collected_package`, generates N deterministic episode
variants (`seed_0000` canonical + instruction variant 0, `seed_0001..N` perturbed):
  - [C] Configuration jitter: +/-2.0 cm XY and +/-12 deg yaw on movable target props.
  - [B] Background/Clutter jitter: +/-3.5 cm XY and +/-20 deg yaw on context props,
        enforcing 2D AABB minimum clearance against all other tabletop objects and
        shifting any `absolute` `object_at_initial_pose` anti-knockover bounds in `goal`
        by the exact `(dx, dy)` delta so per-seed predicates remain exact.
  - [I] Instruction diversity: cycling concise IEEE TCE natural-language prompts.
Outputs:
  - `seed_XXXX/episode_metadata.json`
  - `episodes_batch_spec.json` (ready for `trajectory-forge batch-synthesize --episodes-json`)
  - `instruction_map.json` (ready for `trajectory-forge export-lerobot --instruction-map`)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import random
from typing import Any


SPLIT_MODES: tuple[str, ...] = (
    "train_mixed",
    "test_c_only",
    "test_b_only",
    "test_i_only",
    "test_mixed",
)

INSTRUCTION_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "ce_01_espresso_capsule_brew": (
        "Pull out the espresso machine drip tray, seat the cup under the spout, drop the capsule into the top chute, lock the brew lever down, turn the selector dial clockwise to espresso, and press and hold the dial to brew.",
        "Extend the front stainless drip tray on the coffee maker, position the ceramic cup on the tray, insert the capsule into the chute, close the chrome brew lever interlock, dial to espresso, and press the center encoder.",
    ),
    "ce_01_home_espresso_brew_v2": (
        "Press the espresso machine power key to wake the OLED screen, pull out the stainless drip tray, weave past the kitchen clutter to place the espresso cup under the spout, load the aluminum capsule into the chute, close the brew lever interlock, rotate the selector dial clockwise to the Espresso setting, and press the center dial button to extract coffee while leaving all surrounding kitchenware undisturbed.",
        "Wake the smart espresso center via its top-right power button, extend the drip tray, seat the ceramic cup beneath the twin nozzle, drop a coffee capsule into the brew chamber, lock down the chrome brew lever until the interlock clears, dial to Espresso on the OLED menu, and confirm extraction without knocking over any nearby kitchen items.",
        "Switch on the brushed-steel coffee station using its right-side wake key, draw out the front grate tray, transfer the demitasse cup around nearby jars onto the tray pad, insert the espresso pod from above, pull the brew lever forward to lock, turn the rotary knob to Espresso, and push the center encoder while keeping tabletop props upright.",
        "Activate the smart dual-screen coffee maker by pressing its upper-right button, pull the lower drip basin outward, set the porcelain cup directly under the dual spouts, place the coffee pod into the top bay, close the brew head lever cleanly, twist the selector dial clockwise past Ristretto and Lungo to Espresso, and depress the encoder to brew.",
        "Held-out kitchen instruction A: power on the countertop espresso studio, slide forward the stainless drip base, carefully thread the espresso cup past neighboring tableware onto the brew zone, drop in the capsule, latch the front lever shut, turn the rotary encoder to the Espresso icon, and click the center knob without disturbing surrounding objects.",
        "Held-out kitchen instruction B: initialize the dual-OLED espresso brewer, extend the cup tray, relocate the white-and-gold cup under the dispensing nozzle, load the aluminum coffee pod vertically into the top chute, lock the interlock handle down, dial clockwise to select Espresso, and press the center encoder button while keeping all surrounding clutter intact.",
    ),
    "ce_02_home_steam_oven_reheat_v2": (
        "Press the convection oven WAKE button, pull open the oven door and extend the wire rack, load the ceramic baking cocotte onto the rack without spilling, slide the rack home and close the door interlock, open the top steam vent slider, dial the selector to 210C Convection Air Fry, and press START BAKE while keeping all kitchenware undisturbed.",
        "Wake the smart steam oven using its top-left button, swing open the drop-down glass door and pull out the wire shelf, transfer the glazed ceramic cocotte cleanly onto the extended rack, push the shelf back and latch the oven door shut, slide open the roof steam vent, turn the rotary dial to Air Fry 210C, and press START BAKE without bumping surrounding counter items.",
        "Activate the countertop convection steam oven by tapping WAKE, draw the front handle downward to open the door, slide the steel cooking rack outward, nestle the oval ceramic baking dish onto the rack without tilting, push the rack inward, shut the oven door firmly, nudge the steam vent slider open, twist the mode dial clockwise to 210C, and push START BAKE.",
        "Initialize the kitchen steam oven workstation via the WAKE key, hinge open the door and extend the wire tray, lift the ceramic broth cocotte smoothly onto the rack seat, return the rack into the cavity and close the door interlock, open the top vent gate, dial the temperature selector to 210C convection, and confirm with START BAKE while leaving nearby kitchen props untouched.",
        "Held-out kitchen instruction A: power on the countertop convection steamer with WAKE, lower the glass oven door, pull forward the metal baking rack, carry the ceramic cocotte level onto the rack, slide the rack home, latch the door closed, slide the top vent open, rotate the mode knob past steam to Air Fry 210C, and hit START BAKE without disturbing neighboring tableware.",
        "Held-out kitchen instruction B: tap WAKE on the convection oven fascia, open the hinged oven door and draw the rack forward, set the baking cocotte gently inside the pulled shelf without sloshing, retract the shelf and shut the door, open the upper steam slider, set the selector knob clockwise to 210C Air Fry, and press START BAKE while keeping every countertop prop in place.",
    ),
    "ce_03_home_kettle_temp_boil_v2": (
        "Carry the water-filled gooseneck kettle gently onto the heated base socket without spilling, seat the infuser lid straight onto the kettle rim interlock, rotate the temperature dial to 100C, slide KEEP WARM active, and press BOIL while keeping all kitchen items undisturbed.",
        "Lift the filled gooseneck kettle vessel by its loop handle and place it flat onto the precision power base socket, lower the stainless infuser lid onto the kettle mouth until seated, twist the temperature dial clockwise to 100C, slide the left KEEP WARM switch on, and press BOIL without disturbing nearby kitchen clutter.",
        "Transfer the water-charged gooseneck kettle softly onto the kettle dock pad so no water spills, cap the kettle throat cleanly with the removable infuser lid, turn the right rotary dial to the 100C boil setting, engage the front KEEP WARM slider, and tap the rightmost BOIL button cleanly while leaving countertop props untouched.",
        "Move the gooseneck teapot carefully onto the docking heater base with minimal slosh, align and seat the circular infuser lid onto the kettle collar interlock, rotate the temperature knob to 100C, slide KEEP WARM across to its active stop, and confirm heating by pressing BOIL while keeping the study mug and breakfast props undisturbed.",
        "Held-out kitchen instruction A: carry the water kettle level onto the power dock ring, place the infuser lid securely into the top opening to clear the lid interlock, dial clockwise to 100C, shift the KEEP WARM slider rightward, and tap BOIL without knocking adjacent counter items.",
        "Held-out kitchen instruction B: seat the gooseneck kettle cleanly inside its heater socket without spilling, drop the infuser basket lid straight onto the kettle rim, rotate the temperature selector to 100C, turn on KEEP WARM via the front slider, and press BOIL while preserving all surrounding kitchenware positions.",
    ),
    "ce_04_workstation_osd_shutdown": (
        "Confirm shutdown by pressing the keyboard Enter key, press the circular silver power button on the desktop terminal, press the lower-right monitor bezel power switch, and align the wired mouse inside the right inspection zone.",
        "Tap Enter on the wired keyboard to accept the prompt, actuate the desktop terminal front shutdown button, turn off the monitor via its bottom-right bezel key, and square the mouse neatly in its parking zone.",
    ),
    "ce_04_home_workstation_osd_save_shutdown_v2": (
        "Dock the aluminum backup drive into the workstation bay and align the wireless mouse onto the mat pad, press the keyboard Enter key to confirm saving, slide the front IO shutter closed to clear the power interlock, press the tower power button for standby, and turn off the monitor bezel power key while leaving desk props undisturbed.",
        "Seat the portable aluminum backup SSD inside the tower bay pocket, move the wireless mouse cleanly onto its right desk pad, tap Enter on the keyboard to save the open project, slide the front IO shutter closed, actuate the tower power switch, and press the display bezel power button without bumping nearby accessories.",
        "Insert the aluminum backup cartridge into the lower workstation slot, square the compact mouse on its mat dock, press the Enter switch to accept the save dialog, push the protective IO shutter rightward to unlock standby, power down the tower, and switch off the monitor panel cleanly.",
        "Dock the external backup drive into the front bay and park the compact desk mouse on the leather mat, tap keyboard Enter to commit work, slide the IO door closed over the ports, trigger the tower standby key, and turn off the display bezel switch while preserving all surrounding study props.",
        "Held-out study instruction A: load the aluminum drive inside the tower dock bay and align the mouse on the mat pad, press Enter to confirm file saving, close the sliding front IO shutter, press the circular tower power switch, and press the lower-right monitor power key without disturbing desk clutter.",
        "Held-out study instruction B: seat the backup SSD neatly into the workstation bay and position the wireless mouse over its mat docking zone, hit Enter on the desktop keyboard, slide the IO bay shutter closed, tap the tower standby button, and power down the monitor screen cleanly.",
    ),
    "ce_05_home_tb_dock_nvme_ingest_v2": (
        "Power on the Thunderbolt RAID dock via HOST POWER, open the front NVMe bay door, insert the aluminum NVMe SSD cartridge into its left bay seat and close the bay door, open the top SD slot gate, dock the SD card vertically into its right reader seat, turn the mode dial clockwise to EXTEND, and press INGEST while leaving study accessories undisturbed.",
        "Tap HOST POWER to wake the aluminum Thunderbolt RAID hub, swing open the front SSD door, load the NVMe cartridge into the left slot and shut the bay door, flip open the top SD reader cover, insert the SD card upright into its reader slot, rotate the right mode dial clockwise to EXTEND, and press INGEST cleanly.",
        "Switch on the desktop RAID ingest station using HOST POWER, pull open the left drive door, push the NVMe SSD into its horizontal slot and close the door flush, lift the SD protective flap, seat the SD media card vertically in the slot, turn the mode dial clockwise past mirror to EXTEND, and hit INGEST without disturbing neighboring clutter.",
        "Activate the Thunderbolt dock via its left HOST POWER key, hinge open the bay cover, insert the aluminum NVMe sled and latch the front door shut, hinge open the top SD gate, dock the SD card upright into the slot, dial the rotary mode selector clockwise to EXTEND, and depress INGEST.",
        "Held-out study instruction A: power up the RAID ingest dock with HOST POWER, swing open the drive bay cover, slide in the NVMe SSD cartridge and reclose the door, flip up the SD slot gate, stand the SD card securely in the vertical reader, rotate the mode dial clockwise to EXTEND, and press INGEST cleanly.",
        "Held-out study instruction B: press HOST POWER on the Thunderbolt dock, open the NVMe front door, seat the SSD module inside and shut the bay cover, lift open the top SD flap, insert the SD card vertically, twist the selector knob clockwise to EXTEND, and confirm copy by pressing INGEST without touching study props.",
    ),
    "ce_06_home_ultrabook_stand_connect_v2": (
        "Raise the laptop docking stand lid open, elevate the riser tilt lock lever, dock the magnetic USB-C PD power plug into the left port and the HDMI adapter dongle into the right port until latched, and tap the keyboard CONFIRM key while keeping surrounding desk accessories undisturbed.",
        "Swing open the laptop stand clamshell screen, lift the ergonomic riser tilt lock upward, mate the left magnetic PD power connector and right HDMI adapter dongle cleanly into the dual side guides, and press the CONFIRM switch on the keyboard deck cleanly.",
        "Lift the ultrabook lid on the riser hub open, pull the tilt lever up to elevate the stand angle, insert the USB-C power plug into the left port guide and the HDMI dongle into the right port guide until seated, and tap CONFIRM without knocking nearby desk props.",
        "Open the laptop docking display fully, raise the left riser tilt arm into its locked detent, snap the left magnetic power plug and right display dongle flush into their side ports, and press the CONFIRM button while leaving all study clutter untouched.",
        "Held-out study instruction A: unlatch and open the ultrabook screen on the creator stand, raise the riser tilt lever, plug the magnetic PD connector into the left guide socket and the HDMI adapter into the right guide socket, and depress CONFIRM neatly.",
        "Held-out study instruction B: raise the hinged laptop lid open, tilt the creator docking riser upward until locked, seat the USB-C PD cable end into the left port and the HDMI dongle into the right port, and tap keyboard CONFIRM while keeping surrounding desk items in place.",
    ),
    "ce_07_home_monitor_riser_privacy_tidy_v2": (
        "Pull open the monitor riser storage drawer, stow the wireless presenter pen inside the tray and close the drawer flush, place the compact mouse onto the right charging pad, slide the webcam privacy shutter closed, turn the brightness dial clockwise, and press the eco sleep key while leaving study accessories undisturbed.",
        "Slide out the front storage drawer beneath the smart monitor riser, lay the presenter pen inside its tray pocket and push the drawer closed, set the wireless mouse onto the right inductive pad, slide the upper-left webcam shutter rightward closed, rotate the brightness knob clockwise, and tap SLEEP cleanly.",
        "Draw forward the aluminum riser organizer drawer, place the sleek presenter pen neatly inside and slide the drawer shut, center the desk mouse onto the inductive charge mat, pull the webcam privacy cover closed, dial brightness upward clockwise, and press the SLEEP button without disturbing nearby clutter.",
        "Open the smart monitor stand front drawer, tuck the wireless presenter pen inside and close the drawer completely, dock the compact mouse on its charging pad to activate 15W charging, slide the privacy block over the camera lens, turn brightness up clockwise, and press SLEEP.",
        "Held-out study instruction A: extend the monitor riser storage drawer, stow the presenter pen inside and push the drawer flush closed, align the compact mouse on the right charging platform, slide the camera privacy gate closed, twist the brightness dial clockwise, and tap the front SLEEP key cleanly.",
        "Held-out study instruction B: pull the riser drawer open, put away the wireless presenter pen and reclose the drawer, set the mouse on its top-right inductive pad, slide the privacy shutter shut, turn the rotary brightness dial clockwise, and activate Eco Standby via SLEEP without bumping tabletop accessories.",
    ),
    "ce_08_home_printer_scan_copy_setpoint_v2": (
        "Push the open paper cassette closed, raise the flatbed scanner lid, lay the optical calibration sheet onto the glass platen, close the scanner lid interlock, tilt the control panel upright, dial the copy counter to 1 copy, and press START while keeping all study accessories undisturbed.",
        "Slide the extended front paper tray completely flush into the laser MFP, swing up the flatbed scanner cover, transfer the calibration sheet flat onto the glass scan bed, shut the scanner lid firmly, lift the front control panel tilt angle, turn the rotary copy knob to 1 copy, and hit START.",
        "Close the open bottom cassette drawer on the multifunction printer, open the upper scanner lid, place the A4 calibration target cleanly on the platen glass, lower the scanner cover shut, angle the console panel upward, dial copy count counterclockwise to 01, and press the right START key cleanly.",
        "Push the protruding paper cassette back into the printer chassis, lift the hinged flatbed cover, seat the optical calibration card onto the glass bed, close the cover interlock, adjust the control fascia tilt upward, set the copy selector to 1 copy, and tap START without disturbing desk props.",
        "Held-out study instruction A: shut the MFP paper tray flush, hinge open the scanner lid, position the optical calibration sheet onto the platen surface, close the lid neatly, tilt up the control console, rotate the copy dial to 1 copy, and press START.",
        "Held-out study instruction B: push the lower paper drawer home, lift the top scanner cover, transfer the calibration target card onto the glass platen, close the cover to clear the interlock, tilt the operator panel upright, set the copy count dial to 01, and depress START while preserving nearby desk clutter.",
    ),
    "ce_09_home_printer_duplex_clear_standby_v2": (
        "Open the MFP scanner lid and close it back shut, transfer the optical sheet neatly into the right walnut document tray, flip open the front output stopper, push the lower paper cassette closed flush, and press the MODE button to put the printer into Eco Standby while leaving desk accessories undisturbed.",
        "Raise the scanner cover on the laser MFP, relocate the optical target sheet into the walnut inbox tray beside the printer, close the scanner lid cleanly, unfold the front paper output stopper, slide the open paper cassette shut, and tap MODE for 0.4W Eco Standby.",
        "Check inside by opening the flatbed scanner cover, file the calibration sheet squarely inside the right walnut tray, swing the scanner lid closed, flip down the output catcher flap, push the lower cassette tray home, and press the MODE switch to enter Eco Sleep without knocking desk items.",
        "Hinge up the scanner lid, pick and deposit the A4 calibration sheet into the walnut document holder, lower the scanner lid shut, deploy the front output tray stopper, close the front paper cassette completely, and press MODE cleanly.",
        "Held-out study instruction A: lift open the printer scanner cover, stow the optical calibration card into the right-side walnut inbox tray, close the scanner cover flush, flip the output stopper flap open, slide the paper cassette inward, and press MODE for Eco Standby.",
        "Held-out study instruction B: raise and inspect the MFP flatbed lid, place the printed optical sheet into the walnut document inbox, shut the flatbed cover, deploy the front output stop wing, close the protruding paper tray, and tap MODE while leaving surrounding study props undisturbed.",
    ),
    "ce_08_printer_load_and_copy_setpoint": (
        "Push the extended printer paper cassette completely closed, place the optical calibration sheet flat on the glass platen, close the top scanner lid interlock, and press and hold the green START copy key.",
        "Slide the open front paper drawer smoothly into the multifunction printer chassis, transfer the calibration target sheet onto the flatbed platen, lower the ADF scanner cover shut, and hold START to initiate copying.",
    ),
    "ce_10_home_projector_gimbal_focus_v2": (
        "Press the projector rear POWER button to boot optics, slide open the front lens protective shutter, tilt the gimbal head upward to project, pan gimbal yaw to aim, rotate the front optical focus ring to sharpen projection, and press KEYSTONE LOCK while keeping living console decor undisturbed.",
        "Power on the gimbal laser cinema projector using its rear right switch, push the front lens protective door rightward open, tilt the optical barrel up, pan the gimbal cradle rightward toward the wall, twist the front focus collar clockwise to sharpen, and press KEYSTONE LOCK cleanly.",
        "Boot the tabletop 4K projector via rear POWER, slide back the front shutter leaf to expose the lens, pitch the gimbal barrel upward and swing yaw clockwise into alignment, turn the optical focus dial clockwise, and press the rear left KEYSTONE LOCK button without bumping living room props.",
        "Tap rear POWER on the cinema gimbal projector, slide the lens gate open, elevate gimbal tilt pitch and pan yaw clockwise, calibrate sharpness by twisting the front lens focus collar, and latch alignment with KEYSTONE LOCK while preserving console accessories.",
        "Held-out living room instruction A: activate the cinema projector with the rear right power key, slide the front lens cover open, pitch the projector head upward and pan yaw clockwise, adjust the front focus ring clockwise, and press KEYSTONE LOCK without disturbing surrounding objects.",
        "Held-out living room instruction B: turn on the gimbal 4K projector via rear POWER, slide the optical shutter aside, angle gimbal pitch up and yaw clockwise toward the screen, dial focus clockwise to crisp image, and lock geometry using KEYSTONE LOCK.",
    ),
    "ce_11_home_speaker_privacy_eq_setup_v2": (
        "Tap the speaker top WAKE pad, toggle SOURCE to Studio Radio, rotate the outer volume crown ring clockwise, swing open the rear acoustic bass port flap, and slide the microphone privacy switch to mute while leaving living console props undisturbed.",
        "Touch the left top WAKE sensor on the tube speaker, press SOURCE to switch to Studio Radio, twist the knurled upper volume dial ring clockwise, pull open the rear circular acoustic port flap, and slide the rear microphone privacy gate rightward to mute cleanly.",
        "Wake the acoustic tube smart speaker via its top-left touch pad, cycle SOURCE via the top-right pad, dial the outer brass volume crown clockwise, hinge open the rear bass reflex port door, and push the upper rear MIC MUTE slider across without bumping console clutter.",
        "Initialize the walnut-console tube speaker by tapping WAKE, press SOURCE to select Studio Radio, turn the top perimeter volume crown clockwise, open the back acoustic bass door, and engage MIC MUTE via the rear slider while keeping decor intact.",
        "Held-out living room instruction A: press the speaker top WAKE pad, tap SOURCE to change input, rotate the perimeter volume crown clockwise, pull back the rear bass port cover, and slide the microphone privacy shutter to mute cleanly.",
        "Held-out living room instruction B: activate the smart cylindrical speaker by depressing WAKE, tap SOURCE, turn the top outer ring clockwise for higher gain, open the rear lower acoustic port hinge, and slide mic privacy rightward without disturbing adjacent console props.",
    ),
    "ce_13_home_humidifier_refill_hygrostat_v2": (
        "Hinge open the humidifier top tank lid, pour water smoothly from the bedside carafe over the open rim before returning the carafe, close the top tank lid shut, pull out the aroma drawer to drop in the aroma tablet and close the drawer, set the hygrostat dial counter-clockwise to 45% RH, and press NIGHT MIST and UV STERILIZE while leaving nightstand props undisturbed.",
        "Flip open the ultrasonic humidifier top cover, transfer water gently from the glass bedside carafe into the open reservoir and return the carafe cleanly, lower the top lid closed, slide forward the lower aroma drawer, place the tablet puck inside and shut the drawer, dial the rotary hygrostat counter-clockwise to 45% RH, and tap NIGHT MIST plus UV STERILIZE.",
        "Swing back the top cover of the warm mist humidifier, pour water steadily from the bedside pour carafe into the open water basin and set the carafe back upright, latch the lid down, draw open the bottom aroma tray, seat the essential tablet inside and close the tray, rotate the front hygrostat knob counter-clockwise to 45% RH, and trigger both NIGHT MIST and UV STERILIZE.",
        "Raise the hinged humidifier cap, replenish the water chamber cleanly with the beside carafe before parking it back, shut the top tank cover, pull out the fragrance drawer to insert the aroma tablet puck and reclose it, turn the hygrostat dial counter-clockwise to 45% RH, and press NIGHT MIST and UV STERILIZE while preserving nightstand accessories.",
        "Held-out bedroom instruction A: open the humidifier hinged lid, pour water from the bedside carafe into the reservoir without sloshing and return the carafe, close the cap, pull open the aroma tray, insert the scent tablet and push shut, turn the hygrostat counter-clockwise to 45% RH, and press both NIGHT MIST and UV STERILIZE.",
        "Held-out bedroom instruction B: lift open the top tank cover on the bedroom humidifier, decant water smoothly from the pour carafe and return it beside the unit, close the tank lid, slide out the lower aroma drawer to stow the puck and push flush, twist hygrostat counter-clockwise to 45% RH, and actuate NIGHT MIST and UV STERILIZE.",
    ),
    "ce_14_home_power_station_gan_engage_v2": (
        "Flip open the AC safety cover, plug the three-pin GaN power connector into the front socket, toggle the master surge switch on, press AC1 OUT and USB-C PD switches active, and turn the selector dial to FULL AC+PD 140W while leaving nightstand items undisturbed.",
        "Raise the clear polycarbonate cover over the AC socket, insert the three-pin power plug squarely into the front receptacle, rock the master surge breaker switch on, press AC1 OUT and USB-C PD channel keys, and turn the right mode dial clockwise to FULL AC+PD 140W.",
        "Lift open the hinged safety shield on the GaN desktop power block, mate the three-blade plug into the left front AC outlet, flip the master rocker switch active, tap both AC1 OUT and USB-C PD zone keys, and rotate the selector knob clockwise to 140W full output without bumping nightstand clutter.",
        "Swing up the protective AC shutter, seat the GaN three-pin mains plug into the outlet socket, engage the left MASTER rocker switch, press the center AC1 and USB-C PD output buttons, and rotate the mode dial clockwise to FULL 140W while keeping dresser props undisturbed.",
        "Held-out bedroom instruction A: flip back the transparent AC cover on the GaN power station, push the three-pin plug cleanly into the outlet, switch on the master surge toggle, press AC1 OUT and USB-C PD buttons, and twist the right selector clockwise to FULL 140W.",
        "Held-out bedroom instruction B: open the safety cover above the front socket, insert the three-pin connector home, engage the left master power rocker, activate AC1 OUT and USB-C PD via top push buttons, and turn the mode dial to 140W full delivery without touching surrounding props.",
    ),
    "ce_15_home_kettle_espresso_ristretto_v2": (
        "Dock the water-filled gooseneck kettle onto the left heating base, seat the infuser lid, dial 100C and start boil, then wake the right espresso station, extend its drip tray, place the cup and coffee capsule, lock the brew lever, select Ristretto/Espresso and press brew while leaving kitchen props undisturbed.",
        "Transfer the gooseneck kettle cleanly onto the left thermal base socket, cap the kettle with its infuser lid, dial 100C boil and press BOIL, then pull out the right espresso drip tray, align the espresso cup under the spout, drop a capsule into the top chute, lock the brew handle, rotate the selector to Espresso, and push to extract.",
        "Start by docking the PBD-filled left gooseneck vessel onto its heated base and seating the stainless infuser lid, engage 100C boil on the left dock, and orchestrate the right espresso machine from tray extension through cup placement, capsule load, lever lock, dial selection, and center button extraction.",
        "Orchestrate the dual-appliance kitchen service: mount the kettle on the left dock with its infuser lid and start 100C boiling, then prepare an espresso shot on the right brewer by extending the drip tray, placing the cup and capsule, closing the brew lever, dialing Espresso, and pressing start.",
        "Held-out kitchen instruction A: dock the gooseneck kettle onto the left heated socket and cap the lid to boil at 100C, then pull the right espresso drip tray, load the cup and capsule, close the lever, select Espresso on the front dial, and trigger brewing without knocking kitchen props.",
        "Held-out kitchen instruction B: seat the gooseneck kettle vessel and infuser lid on the left thermal dock, twist the temperature dial to 100C and tap BOIL, then complete Ristretto extraction on the right espresso brewer cleanly while keeping all island accessories undisturbed.",
    ),
    "ce_16_home_tbdock_workstation_ingest_shutdown_v2": (
        "Power on the Thunderbolt RAID dock, open the NVMe bay door to insert and latch the SSD cartridge, switch the dock dial to EXTEND 4K60 and trigger INGEST, then dock the workstation mouse, confirm the OSD save dialog with Enter, close the tower IO shutter, and power off both the tower and monitor while leaving study props undisturbed.",
        "Boot the left Thunderbolt dock, swing open its front drive bay cover, seat the aluminum NVMe cartridge and relatch the bay door, dial EXTEND 4K60 and tap INGEST, then park the study mouse onto the right workstation pad, hit keyboard Enter to clear the OSD prompt, shut the front IO cover, and power down tower and monitor.",
        "Execute the cross-device studio ingest and safe power-down: energize the Thunderbolt RAID dock on the left, load and latch the NVMe cartridge and launch ingest, then switch to the creator workstation on the right to dock the mouse, confirm Enter, close the IO shutter, and turn off both tower and display.",
        "Mount the anodized NVMe drive inside the powered Thunderbolt dock and start EXTEND 4K60 ingestion, then clean up the creator workstation by returning the mouse to the dock pad, pressing Enter on the mechanical keyboard, closing the front port shutter, and tapping tower plus monitor power keys.",
        "Held-out study instruction A: wake the left Thunderbolt RAID hub, install the NVMe SSD into its drive bay and start ingest in EXTEND mode, then dock the workstation mouse, confirm the OSD popup via Enter, slide the IO shutter closed, and power down tower and monitor.",
        "Held-out study instruction B: insert the NVMe cartridge into the left powered Thunderbolt dock and press INGEST at 4K60 EXTEND, then prepare the right workstation for shutdown by parking the mouse, tapping Enter, closing the port shield, and pressing both power keys.",
    ),
    "ce_17_home_printer_copies07_sleep_v2": (
        "Close the MFP paper cassette, raise the scanner lid to seat the optical calibration sheet on the glass platen, close the lid, tilt the control panel upright, dial the copy counter clockwise to the held-out 07 copies setting, press START, and press MODE for Eco Sleep while leaving all study props undisturbed.",
        "Slide the front laser MFP paper tray shut, lift open the flatbed cover to lay the calibration sheet cleanly on glass, close the lid, pivot up the front console, turn the copy count knob clockwise to 07 copies, trigger START, and press MODE to switch to Eco Sleep.",
        "Seat the open MFP paper drawer flush, open the scanner top to place the target card flat onto the platen and close the cover, tilt the tilted fascia panel level, advance the copy dial clockwise to COPIES 07, press START to scan, and enter Eco Sleep with MODE without disturbing study clutter.",
        "Prepare a 7-copy scan on the clean-room laser MFP: push in the bottom cassette tray, load the calibration sheet inside the glass scanner bed, close the scanner cover, level the control panel, twist the copy selector to the unseen 07 setting, tap START, and tap MODE for standby.",
        "Held-out study instruction A: push the paper cassette closed, place the optical target sheet onto the flatbed glass under the closed scanner lid, raise the control fascia, turn the copy dial clockwise past 3 and 5 to COPIES 07, press START, and activate MODE sleep.",
        "Held-out study instruction B: close the study printer paper drawer, insert the calibration sheet onto the scanner platen and lower the lid, tilt up the display panel, rotate the copy quantity dial to 07 copies, execute START scan, and tap MODE for Eco Sleep cleanly.",
    ),
    "ce_18_home_humidifier_rh65_night_uv_v2": (
        "Hinge open the humidifier top tank lid, pour water smoothly from the bedside carafe into the open tank and return the carafe upright, close the lid, load the aroma tablet into the lower tray and shut it, rotate the hygrostat dial clockwise to the unseen 65% RH setpoint, and press NIGHT MIST and UV STERILIZE while leaving all nightstand items undisturbed.",
        "Lift the hinged humidifier cover, transfer water gently from the PBD bedside carafe and place the carafe back on the dresser, shut the top tank cover, pull open the bottom aroma tray to insert the essential puck and push shut, turn the rotary hygrostat clockwise to the held-out 65% RH band, and activate NIGHT MIST and UV STERILIZE.",
        "Open the top tank door of the bedside humidifier, decant water smoothly from the carafe into the open chamber before setting it back, reclose the tank cover, load the aroma tablet into the lower drawer, turn the front hygrostat knob clockwise to 65% RH, and press both NIGHT MIST and UV STERILIZE.",
        "Execute high-humidity sterile mist mode on the bedroom humidifier: flip open the top cap, pour bedside carafe water neatly without spilling, return the carafe, close the cap, install the aroma tablet in the drawer, turn the front dial clockwise to 65% RH, and tap NIGHT MIST and UV STERILIZE.",
        "Held-out bedroom instruction A: refill the opened warm-mist humidifier reservoir from the beside carafe and park the carafe back cleanly, close the tank lid, place the scent tablet into the shut aroma drawer, twist the hygrostat clockwise to the unseen 65% RH target, and press NIGHT MIST and UV STERILIZE.",
        "Held-out bedroom instruction B: swing open the humidifier top lid, replenish tank water smoothly with the pour carafe, latch the lid shut, stow the fragrance puck inside the aroma tray, dial clockwise to 65% RH, and press both NIGHT MIST and UV STERILIZE without moving nearby decor.",
    ),
    "ce_19_home_projector_speaker_cinema_v2": (
        "Power on the cinema projector, open the lens shutter, raise the gimbal head upward, sharpen focus and lock keystone, then tap WAKE and SOURCE on the smart speaker, turn the volume crown clockwise, open the rear bass port, and slide the microphone privacy switch to mute while leaving living room console decor undisturbed.",
        "Orchestrate evening cinema mode across the left projector and right speaker: press projector POWER, retract its lens shutter, elevate the gimbal head, turn the lens focus ring and press KEYSTONE LOCK, then tap WAKE and SOURCE on the smart tube speaker, dial up the top crown, open its rear port, and mute mic privacy.",
        "Activate the living room home theater duo by booting the left gimbal projector, uncovering its optics, pitching upward, adjusting the knurled focus ring and pressing keystone lock, while waking the right tube speaker, toggling optical source, rotating the volume crown, opening the bass port, and muting the mic slider.",
        "Calibrate the left 4K projector (power button, shutter slide, gimbal elevation, focus ring, keystone confirm) and configure the right acoustic tube speaker (wake pad, source button, volume ring, acoustic flap pull, privacy mute slide) with zero disruption to console accessories.",
        "Held-out living room instruction A: boot the left gimbal projector, slide the front shutter open, pitch the head upward, tune focus and lock keystone, then initialize the right smart speaker by tapping WAKE and SOURCE, twisting the volume ring clockwise, opening the bass port, and muting mic privacy.",
        "Held-out living room instruction B: configure the dual living-console cinema setup by powering and aligning the left laser projector first, followed by waking the right speaker, setting SOURCE, dialing volume clockwise, unlatching the rear port, and engaging the hardware mic mute.",
    ),
    "ce_20_home_power_station_uv_dock_cascade_v2": (
        "Open the GaN power station safety cover, insert the three-pin plug, switch on master surge, AC1, and USB-C PD to 140W mode, then power and eject the wireless UV dock, dock the smartphone and earbuds case, close the UV hood shut, and press BOOST while leaving nightstand accessories undisturbed.",
        "Energize the bedroom bedside charging cascade: flip back the GaN power station cover, plug the three-pin connector into AC1, turn on the master surge toggle, enable AC1 and USB-C PD buttons and dial FULL 140W, then switch on the wireless UV dock, pop its hood, dock the phone and earbuds, close the lid, and tap BOOST.",
        "Boot the left GaN desktop block (safety lid, three-pin plug mate, master switch, AC1+PD buttons, 140W dial) to energize the right wireless UV sanitizer, then wake PWR and EJECT, align the smartphone on the left 15W coil and the earbuds in the right UV well, latch the cover, and engage BOOST.",
        "Perform full bedroom power cascade setup: insert the GaN plug beneath the lifted cover, enable master breaker and AC1/USB-C outputs at 140W, then wake the dual wireless dock, eject its UV chamber, seat both devices neatly inside, close the smoked hood, and trigger BOOST.",
        "Held-out bedroom instruction A: open the GaN station protective shield, plug the three-blade AC connector home, engage master surge plus AC1 and USB-C PD buttons with the dial at 140W, then wake and eject the UV dock, align phone and earbuds, seal the hood, and press BOOST.",
        "Held-out bedroom instruction B: connect mains power via the left GaN block and dial FULL AC+PD 140W delivery, then power up the right dual wireless charger, open its UV cover, place phone on coil and earbuds inside the cavity, close the visor lid shut, and hit BOOST cleanly.",
    ),
    "ce_12_qi_dual_charger_align": (
        "Press the left power push-button on the dual-coil wireless dock, center and align the smartphone over the 15W left coil ring, place the earbuds charging case onto the 5W right coil ring, and press the BOOST button.",
        "Switch on the dual wireless charging pad via the left power key, dock the titanium smartphone straight onto the left magnetic ring, seat the ceramic earbuds case on the right coil pad, and activate BOOST charging.",
    ),
    "ce_12_home_wireless_uv_dock_v2": (
        "Press the left PWR button to wake the smart bedside charging dock, press the EJECT key to pop open the right UV-C chamber cover, dock the smartphone straight onto the left 15W inductive pad, place the earbuds case inside the open right UV-C bay, push the smoked lid closed until latched for UV sterilization, and press BOOST for 15W+5W fast charging without disturbing nearby nightstand items.",
        "Turn on the dual-coil UV-C bedside station, tap the eject button so the right sterilization lid springs open, align the titanium phone cleanly onto the left magnetic coil, seat the earbuds charging case inside the right UV pod, close the smoked acrylic cover shut, and tap BOOST while keeping surrounding bedroom clutter upright and untouched.",
        "Wake the bedside dual-coil wireless sanitizer using the leftmost PWR switch, actuate the middle EJECT switch to raise the right sanitizing hood, slide the smartphone beneath the visor onto the left 15W coil ring, nest the white earbuds case inside the right UV chamber, press the UV cover flush closed, and trigger the right BOOST switch cleanly.",
        "Initialize the walnut-dresser wireless UV dock via its left power button, press EJECT to open the hinged UV bay lid, seat the compact smartphone flat on the left inductive charger, transfer the earbuds case into the illuminated UV compartment, close the smoked visor cover until latched, and push BOOST without colliding with adjacent dresser accessories.",
        "Held-out bedroom instruction A: activate the smart dual wireless dock by tapping PWR, pop open the right sterilization cover with EJECT, dock the smartphone gently under the display overhang onto the 15W left pad, drop the earbuds case into the right UV tray, seal the lid shut, and press BOOST while leaving all nearby nightstand props undisturbed.",
        "Held-out bedroom instruction B: power up the dual-coil bedside charger station, tap the center EJECT button to lift the UV chamber lid, place the phone flush onto the left magnetic coil, place the earbuds case inside the opened right cavity, press the hinged cover closed to engage UV sterilization, and push the rightmost BOOST key cleanly.",
    ),
}


def _quat_mul_wxyz(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> list[float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return [w / norm, x / norm, y / norm, z / norm]


def _context_prop_ids(task_data: dict[str, Any]) -> set[str]:
    contract = task_data.get("scenario_forge_runtime_contract") or {}
    raw_objs = contract.get("objects")
    out: set[str] = set()
    if isinstance(raw_objs, dict):
        for oid, spec in raw_objs.items():
            if isinstance(spec, dict) and spec.get("role") in {"context_prop", "dynamic_context_object"}:
                out.add(str(oid))
    elif isinstance(raw_objs, list):
        for spec in raw_objs:
            if isinstance(spec, dict) and spec.get("role") in {"context_prop", "dynamic_context_object"}:
                oid = spec.get("id") or spec.get("object_id")
                if oid:
                    out.add(str(oid))
    return out


def _object_footprint_xy(item: dict[str, Any]) -> tuple[float, float]:
    fp = item.get("footprint_xy") or item.get("footprint_xy_m")
    if isinstance(fp, (list, tuple)) and len(fp) >= 2:
        return (float(fp[0]), float(fp[1]))
    scale = item.get("scale")
    sx = float(scale[0]) if isinstance(scale, (list, tuple)) and len(scale) >= 2 else 1.0
    sy = float(scale[1]) if isinstance(scale, (list, tuple)) and len(scale) >= 2 else 1.0
    return (0.08 * sx, 0.08 * sy)


def _clear_of_neighbors(
    *,
    obj_id: str,
    cand_x: float,
    cand_y: float,
    half_x: float,
    half_y: float,
    placed_boxes: dict[str, tuple[float, float, float, float]],
    min_clearance_m: float,
) -> bool:
    for other_id, (ox, oy, ohx, ohy) in placed_boxes.items():
        if other_id == obj_id:
            continue
        gap_x = abs(cand_x - ox) - (half_x + ohx)
        gap_y = abs(cand_y - oy) - (half_y + ohy)
        if max(gap_x, gap_y) < min_clearance_m:
            return False
    return True


def _shift_absolute_goal_bounds(node: Any, deltas_xy: dict[str, tuple[float, float]]) -> None:
    if isinstance(node, list):
        for child in node:
            _shift_absolute_goal_bounds(child, deltas_xy)
        return
    if not isinstance(node, dict):
        return
    if "sub_goal_setting" in node:
        _shift_absolute_goal_bounds(node["sub_goal_setting"], deltas_xy)
        return
    obj_uid = node.get("obj1_uid")
    if isinstance(obj_uid, str) and obj_uid in deltas_xy:
        dx, dy = deltas_xy[obj_uid]
        if node.get("x_type") == "absolute" and isinstance(node.get("x_range"), list) and len(node["x_range"]) == 2:
            node["x_range"] = [round(float(v) + dx, 6) for v in node["x_range"]]
        if node.get("y_type") == "absolute" and isinstance(node.get("y_range"), list) and len(node["y_range"]) == 2:
            node["y_range"] = [round(float(v) + dy, 6) for v in node["y_range"]]


def generate_variants(
    *,
    collected_root: Path,
    out_dir: Path,
    num_episodes: int = 2,
    start_seed: int = 0,
    split_mode: str = "train_mixed",
    export_pkl: bool = False,
    target_xy_jitter_m: float = 0.020,
    target_yaw_jitter_deg: float = 12.0,
    context_xy_jitter_m: float = 0.035,
    context_yaw_jitter_deg: float = 20.0,
    min_clearance_m: float = 0.015,
) -> dict[str, Any]:
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unsupported split_mode {split_mode!r}; expected one of {SPLIT_MODES}")
    resolved_pkg = collected_root.resolve()
    manifest = json.loads((resolved_pkg / "package_manifest.json").read_text(encoding="utf-8"))
    package_id = str(manifest["package_id"])
    ep_rel = manifest["entrypoints"]["episode_metadata"]
    base_doc = json.loads((resolved_pkg / ep_rel).read_text(encoding="utf-8"))
    base_instr = str(base_doc["task_data"].get("instruction", ""))
    instr_pool = INSTRUCTION_PARAPHRASES.get(package_id, (base_instr,))
    if len(instr_pool) >= 6:
        train_instr = instr_pool[:4]
        heldout_instr = instr_pool[4:]
    else:
        train_instr = instr_pool
        heldout_instr = instr_pool
    ctx_ids = _context_prop_ids(base_doc["task_data"])

    task_out = out_dir.resolve() / package_id
    task_out.mkdir(parents=True, exist_ok=True)
    batch_specs: list[dict[str, Any]] = []
    instruction_map: dict[str, str] = {}

    for idx in range(num_episodes):
        seed = start_seed + idx
        rng = random.Random(f"{package_id}:{seed}")
        ep_doc = copy.deepcopy(base_doc)
        task_data = ep_doc["task_data"]
        layout = task_data["initial_layout"]

        should_jitter_scene = (
            split_mode != "test_i_only"
            and not (split_mode == "train_mixed" and idx == 0 and start_seed == 0)
        )
        jitter_target = split_mode in {"train_mixed", "test_c_only", "test_mixed"}
        jitter_context = split_mode in {"train_mixed", "test_b_only", "test_mixed"}

        if should_jitter_scene:
            placed_boxes: dict[str, tuple[float, float, float, float]] = {}
            movable_ids: list[str] = []
            for obj_id, item in layout.items():
                if obj_id in {"00000000000000000000000000000000", "lift2"}:
                    continue
                if not isinstance(item, dict) or item.get("type") != "object":
                    continue
                if "position" not in item:
                    continue
                fp_x, fp_y = _object_footprint_xy(item)
                placed_boxes[obj_id] = (
                    float(item["position"][0]),
                    float(item["position"][1]),
                    0.5 * fp_x,
                    0.5 * fp_y,
                )
                if not item.get("is_articulation_part") and not item.get("is_articulated", False):
                    contract_objs = (task_data.get("scenario_forge_runtime_contract") or {}).get("objects") or []
                    is_fixed_hero = any(
                        isinstance(c, dict)
                        and c.get("id", c.get("object_id", c.get("scenario_object_id"))) == obj_id
                        and c.get("role") == "articulated_appliance"
                        for c in (contract_objs.values() if isinstance(contract_objs, dict) else contract_objs)
                    )
                    if not is_fixed_hero:
                        is_ctx = obj_id in ctx_ids
                        if (is_ctx and jitter_context) or ((not is_ctx) and jitter_target):
                            movable_ids.append(obj_id)

            deltas_xy: dict[str, tuple[float, float]] = {}
            for obj_id in movable_ids:
                item = layout[obj_id]
                is_ctx = obj_id in ctx_ids
                xy_j = context_xy_jitter_m if is_ctx else target_xy_jitter_m
                yaw_j = context_yaw_jitter_deg if is_ctx else target_yaw_jitter_deg
                pos = list(item["position"])
                orig_x, orig_y = float(pos[0]), float(pos[1])
                _, _, hx, hy = placed_boxes[obj_id]

                dx, dy = 0.0, 0.0
                for attempt in range(48):
                    scale = 1.0 if attempt < 32 else 0.5
                    cand_dx = rng.uniform(-xy_j * scale, xy_j * scale)
                    cand_dy = rng.uniform(-xy_j * scale, xy_j * scale)
                    if _clear_of_neighbors(
                        obj_id=obj_id,
                        cand_x=orig_x + cand_dx,
                        cand_y=orig_y + cand_dy,
                        half_x=hx,
                        half_y=hy,
                        placed_boxes=placed_boxes,
                        min_clearance_m=min_clearance_m,
                    ):
                        dx, dy = cand_dx, cand_dy
                        break

                pos[0] = round(orig_x + dx, 6)
                pos[1] = round(orig_y + dy, 6)
                item["position"] = pos
                placed_boxes[obj_id] = (pos[0], pos[1], hx, hy)
                deltas_xy[obj_id] = (dx, dy)

                dyaw_rad = math.radians(rng.uniform(-yaw_j, yaw_j))
                q0 = tuple(float(v) for v in item.get("orientation", [1.0, 0.0, 0.0, 0.0]))
                qz = (math.cos(0.5 * dyaw_rad), 0.0, 0.0, math.sin(0.5 * dyaw_rad))
                item["orientation"] = [round(v, 7) for v in _quat_mul_wxyz(qz, q0)]

            _shift_absolute_goal_bounds(task_data.get("goal", []), deltas_xy)

        if split_mode in {"test_i_only", "test_mixed"}:
            instruction = heldout_instr[idx % len(heldout_instr)]
        elif split_mode in {"test_c_only", "test_b_only"}:
            instruction = train_instr[0]
        else:
            instruction = train_instr[idx % len(train_instr)]
        task_data["instruction"] = instruction
        episode_id = f"{package_id}_ep_{idx:04d}_seed_{seed}"

        ep_dir = task_out / f"seed_{seed:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_meta_path = ep_dir / "episode_metadata.json"
        ep_meta_path.write_text(json.dumps(ep_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if export_pkl:
            with (ep_dir / "meta_info.pkl").open("wb") as pfh:
                pickle.dump(ep_doc, pfh)

        batch_specs.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "split_mode": split_mode,
                "episode_metadata_relpath": str(ep_meta_path),
                "instruction_override": instruction,
            }
        )
        instruction_map[episode_id] = instruction

    spec_path = task_out / "episodes_batch_spec.json"
    spec_path.write_text(json.dumps(batch_specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    imap_path = task_out / "instruction_map.json"
    imap_path.write_text(json.dumps(instruction_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "package_id": package_id,
        "num_episodes": num_episodes,
        "split_mode": split_mode,
        "episodes_batch_spec": str(spec_path),
        "instruction_map": str(imap_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate C+B+I episode variants for CE-OP Bench.")
    ap.add_argument("--collected-root", action="append", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-episodes", type=int, default=2)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--split-mode", default="train_mixed", choices=SPLIT_MODES)
    ap.add_argument("--export-pkl", action="store_true")
    ap.add_argument("--target-xy-jitter-m", type=float, default=0.020)
    ap.add_argument("--target-yaw-jitter-deg", type=float, default=12.0)
    args = ap.parse_args()

    summaries = []
    merged_instructions: dict[str, str] = {}
    for pkg in args.collected_root:
        res = generate_variants(
            collected_root=pkg,
            out_dir=args.out_dir,
            num_episodes=args.num_episodes,
            start_seed=args.start_seed,
            split_mode=args.split_mode,
            export_pkl=args.export_pkl,
            target_xy_jitter_m=args.target_xy_jitter_m,
            target_yaw_jitter_deg=args.target_yaw_jitter_deg,
        )
        summaries.append(res)
        merged_instructions.update(json.loads(Path(res["instruction_map"]).read_text(encoding="utf-8")))

    merged_path = args.out_dir.resolve() / "merged_instructions.json"
    merged_path.write_text(json.dumps(merged_instructions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": summaries, "merged_instructions": str(merged_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
